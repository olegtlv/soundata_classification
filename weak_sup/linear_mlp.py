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

def train_head_only(model_cls, loader, optimizer, device, epochs=50):
    model_cls.train()
    # keep encoder frozen & stable
    # model_cls.encoder.eval()

    for epoch in range(epochs):
        total = 0
        correct = 0
        total_loss = 0.0

        for batch in loader:
            x = batch["x1"].to(device).float()
            y = batch["label"].to(device).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model_cls(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()

        print(f"epoch {epoch+1:03d}  loss={total_loss/total:.4f}  acc={correct/total:.3f}")


def eval_head_enc(model_cls, loader, device):
    model_cls.eval()

    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device).float()
            y = batch["label"].to(device).long()

            logits = model_cls(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()

    print(f"eval  loss={total_loss/total:.4f}  acc={correct/total:.3f}")

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


class LinearHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, z):
        return self.fc(z)

labeled_idx = select_n_per_class(train_ds, n_per_class=3, seed=0, min_conf=None, prefer_high_conf=True)
labeled_loader = DataLoader(
    Subset(train_ds, labeled_idx),
    batch_size=min(cfg.batch_size, len(labeled_idx)),
    shuffle=True,
    pin_memory=True,
)
print("Labeled set size:", len(labeled_idx))

class Classifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        encoder.eval()  # keep it in eval so BN/Dropout don't drift

    def forward(self, x):
        z = self.encoder.encode(x)          # [B, latent_dim]
        z = F.normalize(z, dim=1)           # OK to normalize embeddings
        logits = self.head(z)               # [B, n_classes]
        return logits

head = LinearHead(cfg.latent_dim, n_classes=len(label2id)).to(device)
encoder = ResNetEncoder(latent_dim=cfg.latent_dim, depth="layer2", pretrained=True).to(device)
state = torch.load(r"C:\data\models\ae_resnet_181225.pth", map_location=device)
encoder.load_state_dict(state)

model = Classifier(encoder, head).to(device)
opt = torch.optim.AdamW([
    {"params": model.head.parameters(), "lr": 1e-3},
    {"params": model.encoder.parameters(), "lr": 5e-5},
])
criterion = nn.CrossEntropyLoss()


train_head_only(model, labeled_loader, opt, device, epochs=500)
eval_head_enc(model, test_loader, device)

print(1)
