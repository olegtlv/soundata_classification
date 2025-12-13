# main.py
import torch
from torch.utils.data import DataLoader

from config import Config
from data.data_access import get_train_test_clips_n_labels, create_label_map
from data.dataset import UrbanSoundDataset
from data.augmentations import get_ae_augment, ContrastiveAug, SafeContrastiveAug, StrongContrastiveAug
from models.convAE_model import ConvAutoencoder
from models.convVAE_model import ConvVAE
from models.convVAE_v2 import ConvVAE_v2
from models.convAE_v2 import ConvAutoencoder_v2
from models.resnetEncoder import ResNetEncoder
from models.ConvAE_SE import ConvAutoencoderSE
from models.simCLRModel import SimCLRModel
from training.trainer_ae import AETrainer, VAETrainer
from training.trainer_simclr import SimCLRrainer
from training.embedding import extract_embeddings, extract_embeddings_VAE
from training.clustering_eval import evaluate_clustering, visualize_embeddings_tsne, per_class_cluster_report

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
aug = StrongContrastiveAug()
# aug = get_ae_augment()
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
    model = ConvAutoencoderSE(latent_dim=cfg.latent_dim).to(device)
    # model = ConvAutoencoder_v2(latent_dim=cfg.latent_dim, input_size=(1, 64, 126)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    trainer = AETrainer(model, optimizer, device)
elif cfg.model_type == 'vae':
    model = ConvVAE(latent_dim=cfg.latent_dim).to(device)
    # model = ConvVAE_v2(latent_dim=cfg.latent_dim, input_size=(1, 64, 126)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    trainer = VAETrainer(model, optimizer, device, beta=2.0)  # try stronger KL
elif cfg.model_type == 'contrastive':
    ae = ConvAutoencoder(latent_dim=cfg.latent_dim).to(device)
    ## state = torch.load(r"C:\data\models\audio_ae_v2_281125.pth", map_location=device)
    ## ae.load_state_dict(state)
    ## ae = ConvAutoencoder_v2(latent_dim=cfg.latent_dim, input_size=(1, 64, 126)).to(device)
    state = torch.load(r"C:\data\models\audio_ae_121225_128_params.pth", map_location=device)
    ae.load_state_dict(state)

    # freeze(ae.enc1)
    # freeze(ae.enc2)
    resencoder = ResNetEncoder(latent_dim=cfg.latent_dim, depth="layer2", pretrained=True).to(device)
    model = SimCLRModel(ae_backbone=resencoder, latent_dim=cfg.latent_dim).to(device)
    # load AE weights into the encoder part
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": cfg.lr/3},
        {"params": model.projector.parameters(), "lr": cfg.lr},
    ])
    trainer = SimCLRrainer(model, optimizer, device, temperature=0.3)

warmup_epochs = 5
def warmup_lr(epoch, warmup_epochs=warmup_epochs):
    return min(1.0, (epoch + 1) / warmup_epochs)

warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.epochs-warmup_epochs)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ======================================================
# Train
# ======================================================
for epoch in range(cfg.epochs):
    # train_loss = trainer.train_with_contrastive(train_loader)
    train_loss = trainer.train_epoch(train_loader, epoch)

    if epoch < warmup_epochs:
        warmup.step()
    else:
        scheduler.step()

    print(f"[Epoch {epoch+1}] LR = {scheduler.get_last_lr()[0]:.6f}, Loss = {train_loss:.4f}")

    if epoch % 10 ==0:
        if cfg.model_type == 'ae':
            Z, labels = extract_embeddings(model, test_loader, device)
        elif cfg.model_type == 'vae':
            Z, labels = extract_embeddings_VAE(model, test_loader, device)
        elif cfg.model_type == 'contrastive':
            Z, labels = extract_embeddings(model.encoder, test_loader, device, normalize=True)
        nmi, ari, preds, all_results  = evaluate_clustering(Z,
                                              labels,
                                              base_k=len(label2id),
                                              method="kmeans",
                                              k_multipliers=(1, )
                                              )
        print("Final NMI =", nmi, "Final ARI =", ari)


if cfg.model_type == 'contrastive':
    torch.save(model.encoder.state_dict(), r"C:\data\models\ae_resnet_121225_2l.pth")
else:
    torch.save(model.state_dict(), r"C:\data\models\audio_ae_061225_5.pth")

# ======================================================
# Embeddings + Clustering
# ======================================================
if cfg.model_type == 'ae':
    Z, labels = extract_embeddings(model, test_loader, device)
elif cfg.model_type == 'vae':
    Z, labels = extract_embeddings_VAE(model, test_loader, device)
elif cfg.model_type == 'contrastive':
    Z, labels = extract_embeddings(model.encoder, test_loader, device, normalize=True)

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
