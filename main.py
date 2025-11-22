# main.py
import torch
from torch.utils.data import DataLoader

from config import Config
from data.data_access import get_train_test_clips_n_labels, create_label_map
from data.dataset import UrbanSoundDataset
from data.augmentations import get_ae_augment, ContrastiveAug, SafeContrastiveAug
from models.convAE_model import ConvAutoencoder
from models.convVAE_model import ConvVAE
from models.simCLRModel import SimCLRModel
from training.trainer_ae import AETrainer, VAETrainer
from training.trainer_simclr import SimCLRrainer
from training.embedding import extract_embeddings, extract_embeddings_VAE
from training.clustering_eval import evaluate_clustering, visualize_embeddings_tsne

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

train_ds = UrbanSoundDataset(
    clips=train_clips,
    label2id=label2id,
    precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
    transform=SafeContrastiveAug(),
    mode=cfg.model_type,
    folds=[0,1,2,3,4,5,6,7,8,9]
)

test_ds = UrbanSoundDataset(
    clips=test_clips,
    label2id=label2id,
    precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
    transform=None,
    mode='ae',
    folds=[10]
)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

# ======================================================
# Model + Trainer
# ======================================================

if cfg.model_type == 'ae':
    model = ConvAutoencoder(latent_dim=cfg.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    trainer = AETrainer(model, optimizer, device)
elif cfg.model_type == 'vae':
    model = ConvVAE(latent_dim=cfg.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    trainer = VAETrainer(model, optimizer, device, beta=2.0)  # try stronger KL
elif cfg.model_type == 'contrastive':
    contrastive_transform = ContrastiveAug(target_frames=cfg.target_frames)
    train_ds = UrbanSoundDataset(
        clips=train_clips,
        label2id=label2id,
        precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
        transform=None,
        contrastive_transform=contrastive_transform,
        mode="contrastive",
        folds=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    ae = ConvAutoencoder(latent_dim=cfg.latent_dim).to(device)
    state = torch.load(r"C:\data\models\audio_AE_151125aug_norm_50e", map_location=device)
    ae.load_state_dict(state)

    model = SimCLRModel(ae_backbone=ae, latent_dim=cfg.latent_dim).to(device)
    # load AE weights into the encoder part
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.projector.parameters(), "lr": 1e-4},
    ])
    trainer = SimCLRrainer(model, optimizer, device, temperature=0.1)

def warmup_lr(epoch):
    return min(1.0, (epoch + 1) / 5)

warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.epochs)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ======================================================
# Train
# ======================================================
for epoch in range(cfg.epochs):
    train_loss = trainer.train_epoch(train_loader)

    warmup.step()
    scheduler.step()

    print(f"[Epoch {epoch+1}] LR = {scheduler.get_last_lr()[0]:.6f}, Loss = {train_loss:.4f}")


torch.save(model.state_dict(), r"C:\data\models\audio_SimCLR_221125")
# ======================================================
# Embeddings + Clustering
# ======================================================
if cfg.model_type == 'ae':
    Z, labels = extract_embeddings(model, test_loader, device)
elif cfg.model_type == 'vae':
    Z, labels = extract_embeddings_VAE(model, test_loader, device)
elif cfg.model_type == 'contrastive':
    Z, labels = extract_embeddings(model.encoder, test_loader, device)

nmi, ari, preds = evaluate_clustering(Z, labels, n_clusters=len(label2id))
# visualize_embeddings_tsne(model, test_loader, r"C:\tmp\encoded_tsne1.html", perplexity=12)

print("Final NMI =", nmi)
print("Final ARI =", ari)
print(1)
