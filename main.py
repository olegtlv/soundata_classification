# main.py
import torch
from torch.utils.data import DataLoader
from config import Config
from data.data_access import get_train_test_clips_n_labels, create_label_map
from data.dataset import UrbanSoundDataset
from data.augmentations import get_ae_augment, ContrastiveAug, SafeContrastiveAug, StrongContrastiveAug, \
    make_aug_config, UnifiedAug
from models.convAE_model import ConvAutoencoder
from models.convVAE_model import ConvVAE
from models.resnetEncoder import ResNetEncoder
from models.ConvAE_SE import ConvAutoencoderSE
from models.simCLRModel import SimCLRModel
from training.trainer_ae import AETrainer, VAETrainer
from training.trainer_simclr import SimCLRrainer
from training.embedding import extract_embeddings, extract_embeddings_VAE, pca_transform
from training.clustering_eval import evaluate_clustering, visualize_embeddings_tsne, per_class_cluster_report, \
    summarize_clustering_quality
from models.byol_model import BYOLModel
from training.trainer_byol import BYOLTrainer
from models.vicreg_model import VICRegModel
from training.trainer_vicreg import VICRegTrainer
# ======================================================
# Load config
# ======================================================
cfg = Config()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# Dataset
# ======================================================
train_clips, test_clips, labels = get_train_test_clips_n_labels()
label2id = create_label_map(train_clips + test_clips)

# train_ds = UrbanSoundDataset(
#     clips=train_clips,
#     label2id=label2id,
#     mode="ae",
#     transform=get_ae_augment() if cfg.use_augment else None,
#     use_cache=False,
# )
#
# test_ds = UrbanSoundDataset(
#     clips=test_clips,
#     label2id=label2id,
#     mode="ae",
#     use_cache=False,
# )
# if cfg.model_type == 'contrastive':
# aug = SafeContrastiveAug()
# aug = StrongContrastiveAug()
# aug = get_ae_augment() if cfg.model_type in ['ae', 'vae'] else SafeContrastiveAug() #SafeContrastiveAug() #StrongContrastiveAug()
aug = UnifiedAug(make_aug_config(mode=cfg.mode))

train_ds = UrbanSoundDataset(
    clips=train_clips,
    label2id=label2id,
    precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
    transform=aug,
    mode='contrastive',  # cfg.model_type,
    folds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)

# else:
#     aug = get_ae_augment()
#     train_ds = UrbanSoundDataset(
#         clips=train_clips,
#         label2id=label2id,
#         precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
#         transform=aug,
#         mode='contrastive',  # cfg.model_type,
#         folds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#     )

test_ds = UrbanSoundDataset(
    clips=test_clips,
    label2id=label2id,
    precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
    transform=None,
    mode='ae',
    folds=[10]
)

train_loader = DataLoader(train_ds,
                         batch_size=cfg.batch_size,
                         shuffle=True,
                         pin_memory=True,
                          )
test_loader = DataLoader(test_ds,
                         batch_size=cfg.batch_size,
                         shuffle=False,
                         pin_memory=True,
                         )

# ======================================================
# Model + Trainer
# ======================================================

# ---- FREEZE early encoder layers ----
def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

if cfg.model_type == 'ae':
    model = ConvAutoencoder(latent_dim=cfg.latent_dim).to(device)
    # model = ConvAutoencoder_v2(latent_dim=cfg.latent_dim, input_size=(1, 64, 126)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    trainer = AETrainer(model, optimizer, device)
elif cfg.model_type == 'vae':
    model = ConvVAE(latent_dim=cfg.latent_dim).to(device)
    # model = ConvVAE_v2(latent_dim=cfg.latent_dim, input_size=(1, 64, 126)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    trainer = VAETrainer(model, optimizer, device, beta=2.0)  # try stronger KL
elif cfg.model_type == 'contrastive':
    # ae = ConvAutoencoder(latent_dim=cfg.latent_dim).to(device)
    ## state = torch.load(r"C:\data\models\audio_ae_v2_281125.pth", map_location=device)
    ## ae.load_state_dict(state)
    ## ae = ConvAutoencoder_v2(latent_dim=cfg.latent_dim, input_size=(1, 64, 126)).to(device)
    # state = torch.load(r"C:\data\models\audio_ae_241225.pth", map_location=device)
    # ae.load_state_dict(state)

    # freeze(ae.enc1)
    # freeze(ae.enc2)
    resencoder = ResNetEncoder(latent_dim=cfg.latent_dim, depth="layer2", pretrained=cfg.pretrained).to(device)
    model = SimCLRModel(ae_backbone=resencoder, latent_dim=cfg.latent_dim).to(device)
    # load AE weights into the encoder part
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": cfg.lr / (6 if cfg.pretrained else 3)},
        {"params": model.projector.parameters(), "lr": cfg.lr},
    ])
    trainer = SimCLRrainer(model, optimizer, device, temperature=0.5)
elif cfg.model_type=='byol':
    res_encoder = ResNetEncoder(latent_dim=cfg.latent_dim, depth="layer2", pretrained=True).to(device)
    state = torch.load(r"C:\data\models\ae_byol_181225.pth", map_location=device)
    res_encoder.load_state_dict(state, strict=False)
    # ae_encoder = ConvAutoencoder(latent_dim=cfg.latent_dim).to(device)
    # state = torch.load(r"C:\data\models\ae_sim_2341225.pth", map_location=device)
    # ae_encoder.load_state_dict(state)
    model = BYOLModel(
        encoder=res_encoder,
        latent_dim=cfg.latent_dim,
        proj_dim=128,
        proj_hidden=256,
        pred_hidden=256,
        ema_m=0.996,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    trainer = BYOLTrainer(model, optimizer, device, total_epochs=cfg.epochs, ema_base=0.996)

warmup_epochs = 5
def warmup_lr(epoch, warmup_epochs=warmup_epochs):
    return min(1.0, (epoch + 1) / warmup_epochs)

warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.epochs-warmup_epochs)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ======================================================
# Train
# ======================================================
purity_all = []
nmi_10 = []
nmi_30 = []
for epoch in range(cfg.epochs):
    # train_loss = trainer.train_with_contrastive(train_loader)
    train_loss = trainer.train_epoch(train_loader, epoch)

    # if epoch < warmup_epochs:
    #     warmup.step()
    # else:
    scheduler.step()

    print(f"[Epoch {epoch+1}] LR = {scheduler.get_last_lr()[0]:.6f}, Loss = {train_loss:.4f}")

    if epoch % 10 ==0:
        if cfg.model_type == 'ae':
            Z, labels = extract_embeddings(model, test_loader, device)
        elif cfg.model_type == 'vae':
            Z, labels = extract_embeddings_VAE(model, test_loader, device)
        elif cfg.model_type == 'contrastive':
            Z, labels = extract_embeddings(model.encoder, test_loader, device, normalize=False)
        elif cfg.model_type == 'byol':
            # Z, labels = extract_embeddings(model.target_encoder, test_loader, device, normalize=True)
            Z, labels = extract_embeddings(model.online_encoder, test_loader, device, normalize=False)

        Z_pca = pca_transform(Z, n_components=24, whiten=True, l2_after=True)

        _, _, _, all_results = evaluate_clustering(
            Z_pca,
            labels,
            base_k=len(label2id),
            method="kmeans",
            k_multipliers=(1, 2, 3, 4),
        )

        # choose best k under a cap (e.g. ≤30 clusters)
        MAX_K = 30
        best = None
        best_score = -1

        for k, res in all_results.items():
            if k > MAX_K:
                continue
            summary = summarize_clustering_quality(labels, res["preds"])
            score = summary["weighted_purity"] * summary["frac≥0.70"]
            if score > best_score:
                best_score = score
                best = (k, summary)

        k, s = best
        print(
            f"[k={k:2d}] "
            f"purity_w={s['weighted_purity']:.3f} | "
            f"frac≥0.70={s['frac≥0.70']:.3f} | "
            f"frac≥0.80={s['frac≥0.80']:.3f} | "
            f"eff_cls={s['effective_clusters']:.2f}"
        )
        for k, res in all_results.items():
            print(f"k={k:3d}  NMI={res['nmi']:.3f}  ARI={res['ari']:.3f}")
        purity_all.append(s['weighted_purity'])
        nmi_10.append(all_results[10])
        nmi_30.append(all_results[30])

if cfg.model_type in ['contrastive']:
    torch.save(model.encoder.state_dict(), r"C:\data\models\simclr_res128_241225.pth")
elif cfg.model_type in ['byol']:
    torch.save(model.online_encoder.state_dict(), r"C:\data\models\byol_res128_241225.pth")
else:
    torch.save(model.state_dict(), r"C:\data\models\audio_ae64_241225.pth")

# ======================================================
# Embeddings + Clustering
# ======================================================
if cfg.model_type == 'ae':
    Z, labels = extract_embeddings(model, test_loader, device)
elif cfg.model_type == 'vae':
    Z, labels = extract_embeddings_VAE(model, test_loader, device)
elif cfg.model_type == 'contrastive':
    Z, labels = extract_embeddings(model.target_encoder, test_loader, device, normalize=True)


# nmi, ari, preds = evaluate_clustering(Z, labels, n_clusters=len(label2id)*3)
visualize_embeddings_tsne(model.encoder, test_loader, r"C:\tmp\encoded_tsne2.html", perplexity=12)

id2label = {v: k for k, v in label2id.items()}
per_class_cluster_report(labels, preds, id2label=id2label)


nmi, ari, preds, all_results = evaluate_clustering(
    Z,
    labels,
    base_k=len(label2id),
    method="kmeans",
    k_multipliers=(1, 2, 3, 4, 5,6),
)
for k, res in all_results.items():
    print(f"k={k:3d}  NMI={res['nmi']:.3f}  ARI={res['ari']:.3f}")

print(1)
