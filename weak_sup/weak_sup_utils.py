# main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.embedding import extract_embeddings, extract_embeddings_VAE
from training.clustering_eval import evaluate_clustering, visualize_embeddings_tsne, per_class_cluster_report

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

class LinearHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, z):
        return self.fc(z)


def train_head_only(model_cls, loader, optimizer, device, criterion, model, label2id, test_loader, epochs=50):
    model_cls.train()
    # keep encoder frozen & stable
    model_cls.encoder.eval()

    for epoch in range(epochs):
        if epoch > 40:
            model_cls.encoder.train()
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

        if epoch %20 ==0:
            print(f"epoch {epoch + 1:03d}  loss={total_loss / total:.4f}  acc={correct / total:.3f}")

            Z, labels = extract_embeddings(model.encoder, test_loader, device, normalize=True)
            nmi, ari, preds, all_results = evaluate_clustering(Z,
                                                               labels,
                                                               base_k=len(label2id),
                                                               method="kmeans",
                                                               k_multipliers=(1, 2, 3)
                                                               )
            for k, res in all_results.items():
                print(f"k={k:3d}  NMI={res['nmi']:.3f}  ARI={res['ari']:.3f}")


def eval_head_enc(model_cls, loader, criterion, device):
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


class MetricProjector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=1)

def supervised_contrastive_loss(z, y, temperature=0.2):
    """
    z: [B, D] normalized embeddings
    y: [B] labels
    """
    sim = torch.matmul(z, z.T) / temperature        # [B,B]
    sim = sim - torch.max(sim, dim=1, keepdim=True)[0]  # stability

    y = y.view(-1, 1)
    mask = (y == y.T).float()
    mask.fill_diagonal_(0)

    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1, keepdim=True)

    log_prob = sim - torch.log(denom + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    return -mean_log_prob_pos.mean()

def align_anchors(encoder, projector, loader, device, opt, epochs=20):
    encoder.eval()
    projector.train()

    for ep in range(epochs):
        for batch in loader:
            x = batch["x"].to(device).float()
            y = batch["label"].to(device).long()

            z = encoder.encode(x)
            z = F.normalize(z, dim=1)
            zp = projector(z)

            loss = supervised_contrastive_loss(zp, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if ep % 5 == 0:
            print(f"[align] epoch {ep:02d} loss={loss.item():.4f}")


