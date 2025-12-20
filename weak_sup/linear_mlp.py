# main.py
import torch
import torch.nn as nn
from config import Config
from data.data_access import get_train_test_clips_n_labels, create_label_map
from data.dataset import UrbanSoundDataset
from data.augmentations import get_ae_augment, ContrastiveAug, SafeContrastiveAug, StrongContrastiveAug
from models.convAE_model import ConvAutoencoder
from models.resnetEncoder import ResNetEncoder
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from weak_sup.get_labels import select_n_per_class
from training.embedding import extract_embeddings, extract_embeddings_VAE
from training.clustering_eval import evaluate_clustering, visualize_embeddings_tsne, per_class_cluster_report
from weak_sup.weak_sup_utils import LinearHead, Classifier, train_head_only, eval_head_enc

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

aug = get_ae_augment() #if cfg.model_type in ['ae', 'vae'] else StrongContrastiveAug()

train_ds = UrbanSoundDataset(
    clips=train_clips,
    label2id=label2id,
    precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
    transform=aug,
    mode='contrastive',  # cfg.model_type,
    folds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)

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



labeled_idx = select_n_per_class(train_ds, n_per_class=3, seed=0, min_conf=None, prefer_high_conf=True)
labeled_loader = DataLoader(
    Subset(train_ds, labeled_idx),
    batch_size=min(cfg.batch_size, len(labeled_idx)),
    shuffle=True,
    pin_memory=True,
)
print("Labeled set size:", len(labeled_idx))


head = LinearHead(cfg.latent_dim, n_classes=len(label2id)).to(device)
encoder = ResNetEncoder(latent_dim=cfg.latent_dim, depth="layer2", pretrained=True).to(device)
state = torch.load(r"C:\data\models\ae_resnet_181225.pth", map_location=device)
encoder.load_state_dict(state)

model = Classifier(encoder, head).to(device)
opt = torch.optim.AdamW([
    {"params": model.head.parameters(), "lr": 1e-3},
    {"params": model.encoder.parameters(), "lr": 2e-5},
])
criterion = nn.CrossEntropyLoss()


train_head_only(model, labeled_loader, opt, device, criterion, model, label2id, test_loader, epochs=181)
eval_head_enc(model, test_loader, criterion, device)

print(1)
