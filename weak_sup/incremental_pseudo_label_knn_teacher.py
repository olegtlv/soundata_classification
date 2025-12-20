# main_supcon_label_expansion.py
# Refactor of your main_knn_label_expansion.py from CE -> Supervised Contrastive (SupCon)
#
# Goal:
#   Start with EXACTLY n_per_class anchors, expand pseudo labels conservatively via prototype teacher,
#   and train a metric projector with SupCon (optionally tiny encoder LR).
#
# What changes vs your CE script:
#   1) Classifier head is replaced with a MetricProjector (MLP) that outputs normalized embeddings
#   2) Training loss is SupCon on (anchors + pseudos), with per-sample weights (anchors=1.0, pseudos=w_pseudo*conf)
#   3) Evaluation remains clustering on encoder/projector embeddings (and optional "prototype head eval" without kmeans)
#
# Notes:
#   - Keep encoder frozen for first rounds to prevent BN drift and confirmation bias
#   - Use transform=None at start; add ContrastiveAug later if stable
#   - Pseudo map is persistent across rounds (label memory)
#
# Assumptions:
#   - UrbanSoundDataset(mode="ae") yields dict with keys {"x", "label"}.
#   - ResNetEncoder has .encode(x) -> [B, latent_dim].
#   - extract_embeddings exists; we'll add extract_embeddings_projected for projector eval.

from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import Config
from data.data_access import get_train_test_clips_n_labels, create_label_map
from data.dataset import UrbanSoundDataset
from data.augmentations import ContrastiveAug, StrongContrastiveAug  # start None; add later
from models.resnetEncoder import ResNetEncoder
from training.embedding import extract_embeddings
from training.clustering_eval import evaluate_clustering

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score


def cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total = len(y_true)
    if total == 0:
        return 0.0

    pur_sum = 0
    for k in np.unique(y_pred):
        idx = np.where(y_pred == k)[0]
        if idx.size == 0:
            continue
        # majority class count
        vals, cnts = np.unique(y_true[idx], return_counts=True)
        pur_sum += cnts.max()
    return float(pur_sum) / float(total)

@torch.no_grad()
def eval_subclass_friendly_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    If subclasses are OK, look at Homogeneity & Purity.
    """
    h = homogeneity_score(y_true, y_pred)
    c = completeness_score(y_true, y_pred)
    v = v_measure_score(y_true, y_pred)
    pur = cluster_purity(y_true, y_pred)
    print(f"[subclass-metrics] H={h:.3f}  C={c:.3f}  V={v:.3f}  Purity={pur:.3f}")
    return {"homogeneity": h, "completeness": c, "v_measure": v, "purity": pur}

def _l2_normalize_np(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def allocate_subprotos_per_class(
    counts_per_class: np.ndarray,
    max_total_subprotos: int = 30,
    min_per_class: int = 1,
    max_per_class: int = 6,
) -> np.ndarray:
    """
    counts_per_class[c] = how many points we can use for class c (anchors+pseudos).
    Returns K_per_class[c] number of subprototypes.
    Simple, stable allocator:
      - start with min_per_class each
      - distribute remaining budget proportional to sqrt(count)
      - cap by max_per_class
    """
    C = len(counts_per_class)
    K = np.full(C, min_per_class, dtype=np.int64)

    budget = max_total_subprotos - int(K.sum())
    if budget <= 0:
        return K

    # weights: sqrt(count) favors classes with more confident points but not too aggressively
    w = np.sqrt(np.maximum(counts_per_class, 0).astype(np.float32))
    if w.sum() <= 0:
        return K

    # iterative greedy distribution (small C, stable)
    for _ in range(budget):
        # pick class with largest "need" that isn't at cap
        scores = w / (K.astype(np.float32) + 1.0)
        scores[K >= max_per_class] = -1.0
        c = int(scores.argmax())
        if scores[c] < 0:
            break
        K[c] += 1
    return K

def build_subprototypes(
    Z_all: np.ndarray,              # [N,D] normalized
    y_all: np.ndarray,              # [N]
    labeled_idx: np.ndarray,        # anchors indices
    pseudo_map: dict,               # idx->pseudo_label (persistent)
    pseudo_conf: dict,              # idx->conf (persistent)
    n_classes: int,
    max_total_subprotos: int = 30,
    min_per_class: int = 1,
    max_per_class: int = 6,
    use_only_high_conf_pseudos: bool = True,
    pseudo_conf_thr: float = 0.70,
    seed: int = 0,
):
    """
    Returns:
      P: [M,D] normalized subprototypes
      P_class: [M] class id for each subprototype
    """
    # collect points per class from anchors + (optionally) confident pseudos
    idx_by_class = [[] for _ in range(n_classes)]

    # anchors: always included with true label
    for i in labeled_idx.tolist():
        c = int(y_all[i])
        idx_by_class[c].append(int(i))

    # pseudos: optionally only confident ones (recommended)
    for i, c in pseudo_map.items():
        i = int(i)
        c = int(c)
        if use_only_high_conf_pseudos:
            if float(pseudo_conf.get(i, 0.0)) < float(pseudo_conf_thr):
                continue
        idx_by_class[c].append(i)

    counts = np.array([len(v) for v in idx_by_class], dtype=np.int64)
    K_per_class = allocate_subprotos_per_class(
        counts_per_class=counts,
        max_total_subprotos=max_total_subprotos,
        min_per_class=min_per_class,
        max_per_class=max_per_class,
    )

    protos = []
    proto_classes = []

    rng = np.random.RandomState(seed)

    for c in range(n_classes):
        idxs = idx_by_class[c]
        if len(idxs) == 0:
            continue

        Xc = Z_all[idxs]  # [Nc,D] normalized
        Kc = int(K_per_class[c])

        # if too few points, just one proto
        if Xc.shape[0] < 2 or Kc <= 1:
            mu = Xc.mean(axis=0, keepdims=False)
            protos.append(mu)
            proto_classes.append(c)
            continue

        # kmeans for subclasses in this class
        # n_init small but stable
        km = KMeans(n_clusters=min(Kc, Xc.shape[0]), n_init=5, random_state=rng.randint(0, 10_000))
        lab = km.fit_predict(Xc)

        for k in range(km.n_clusters):
            members = Xc[lab == k]
            if members.shape[0] == 0:
                continue
            mu = members.mean(axis=0)
            protos.append(mu)
            proto_classes.append(c)

    P = np.stack(protos, axis=0).astype(np.float32)
    P = _l2_normalize_np(P)
    P_class = np.array(proto_classes, dtype=np.int64)
    return P, P_class

def subproto_teacher(
    Z_all, labeled_idx, y_all, n_classes,
    pseudo_map, pseudo_conf,
    max_total_subprotos: int = 30,
    min_per_class: int = 1,
    max_per_class: int = 6,
    temp: float = 0.07,
    reject_min_prob: float = 0.35,
    reject_min_margin: float = 0.02,
    pseudo_conf_thr_for_protos: float = 0.70,
    seed: int = 0,
):
    """
    Multi-prototype teacher:
      - builds <= max_total_subprotos subprototypes across classes
      - predicts by softmax over subprototypes
      - maps winning subprototype -> class label
    Returns:
      pred_class [N], conf [N], margin [N], pred_proto_id [N]
    """
    Z = _to_numpy(Z_all).astype(np.float32)
    y = _to_numpy(y_all).astype(np.int64)
    labeled_idx = np.asarray(labeled_idx, dtype=np.int64)

    Zn = _l2_normalize_np(Z)

    # build subprototypes in SAME space as Z (projected space in your script)
    P, P_class = build_subprototypes(
        Z_all=Zn,
        y_all=y,
        labeled_idx=labeled_idx,
        pseudo_map=pseudo_map,
        pseudo_conf=pseudo_conf,
        n_classes=n_classes,
        max_total_subprotos=max_total_subprotos,
        min_per_class=min_per_class,
        max_per_class=max_per_class,
        use_only_high_conf_pseudos=True,
        pseudo_conf_thr=pseudo_conf_thr_for_protos,
        seed=seed,
    )

    # cosine sim to each subprototype
    sim = Zn @ P.T  # [N,M]
    logits = sim / float(temp)
    logits = logits - np.max(logits, axis=1, keepdims=True)

    exp = np.exp(logits)
    proba = exp / (np.sum(exp, axis=1, keepdims=True) + 1e-8)

    # top2 over subprototypes
    top2 = np.partition(proba, -2, axis=1)[:, -2:]
    p1 = np.max(top2, axis=1)
    p2 = np.min(top2, axis=1)
    margin = p1 - p2

    proto_id = np.argmax(proba, axis=1).astype(np.int64)
    pred_class = P_class[proto_id].astype(np.int64)
    conf = p1.astype(np.float32)

    pred_class[(conf < reject_min_prob) | (margin < reject_min_margin)] = -1
    return pred_class, conf, margin, proto_id


# -----------------------------
# Helpers
# -----------------------------
def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# -----------------------------
# Prototype teacher (unchanged)
# -----------------------------
def prototype_teacher(
    Z_all, labeled_idx, y_all, n_classes,
    temp=0.10,
    reject_min_prob=0.35,
    reject_min_margin=0.02,
):
    Z = _to_numpy(Z_all).astype(np.float32)
    y = _to_numpy(y_all).astype(np.int64)
    labeled_idx = np.asarray(labeled_idx, dtype=np.int64)

    # prototypes from anchors
    protos = np.zeros((n_classes, Z.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        idx_c = labeled_idx[y[labeled_idx] == c]
        if idx_c.size == 0:
            continue

        Zc = Z[idx_c]  # [m, D]
        # pick most central anchors: compute distances to mean, keep best half
        mu = Zc.mean(axis=0, keepdims=True)
        d = ((Zc - mu) ** 2).sum(axis=1)
        keep = np.argsort(d)[: max(1, len(d)//2)]
        protos[c] = Zc[keep].mean(axis=0)

    # cosine sim
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    Pn = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    sim = Zn @ Pn.T  # [N, C]

    # softmax conf
    logits = sim / float(temp)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    proba = exp / (np.sum(exp, axis=1, keepdims=True) + 1e-8)

    top2 = np.partition(proba, -2, axis=1)[:, -2:]
    p1 = np.max(top2, axis=1)
    p2 = np.min(top2, axis=1)
    margin = p1 - p2

    pred = np.argmax(proba, axis=1).astype(np.int64)
    conf = p1.astype(np.float32)

    pred[(conf < reject_min_prob) | (margin < reject_min_margin)] = -1
    return pred, conf, margin


def select_pseudo_indices_topN_per_class(
    pred, conf, labeled_set, n_classes,
    top_n=10,
    min_conf=0.65,
    fallback_min_conf=0.45,
    min_per_class=5,
):
    pseudo = []
    per_class_counts = {c: 0 for c in range(n_classes)}

    for c in range(n_classes):
        rows = np.where((pred == c) & (conf >= min_conf))[0]
        if rows.size < min_per_class:
            rows = np.where((pred == c) & (conf >= fallback_min_conf))[0]
        if rows.size == 0:
            continue

        rows = rows[np.argsort(-conf[rows])]
        added = 0
        for r in rows:
            if int(r) in labeled_set:
                continue
            pseudo.append(int(r))
            added += 1
            if added >= top_n:
                break
        per_class_counts[c] = added

    return pseudo, per_class_counts


def pseudo_weight(round_id: int) -> float:
    # conservative ramp
    return min(0.3, 0.05 + 0.05 * round_id)  # 0.10,0.15,0.20,0.25,0.30,...


# -----------------------------
# Dataset mixing true + pseudo
# -----------------------------
class PseudoLabelDataset(Dataset):
    """
    base_ds returns {"x":..., "label":...} with TRUE label.
    For pseudo indices we override label using pseudo_map and add weight, optionally modulated by confidence.

    Returns: {"x": x, "label": y, "weight": w, "is_pseudo": 0/1}
    """
    def __init__(
        self,
        base_ds,
        labeled_idx: List[int],
        pseudo_map: Dict[int, int],
        pseudo_conf: Optional[Dict[int, float]] = None,
        transform=None,
        pseudo_w: float = 0.1,
        conf_power: float = 1.0,   # weight multiplier = conf**conf_power
    ):
        self.base = base_ds
        self.transform = transform

        self.labeled_set = set(int(i) for i in labeled_idx)
        self.pseudo_map = {int(k): int(v) for k, v in pseudo_map.items()}

        self.pseudo_conf = None
        if pseudo_conf is not None:
            self.pseudo_conf = {int(k): float(v) for k, v in pseudo_conf.items()}

        # indices for training
        self.indices = list(self.labeled_set) + list(self.pseudo_map.keys())

        self.pseudo_w = float(pseudo_w)
        self.conf_power = float(conf_power)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = self.indices[j]
        item = self.base[i]
        x = item["x"]
        if self.transform is not None:
            x = self.transform(x)

        if i in self.labeled_set:
            return {"x": x, "label": int(item["label"]), "weight": 1.0, "is_pseudo": 0}
        else:
            y = int(self.pseudo_map[i])
            w = self.pseudo_w
            if self.pseudo_conf is not None and i in self.pseudo_conf:
                w = w * (self.pseudo_conf[i] ** self.conf_power)
            return {"x": x, "label": y, "weight": float(w), "is_pseudo": 1}


# -----------------------------
# Model: encoder + metric projector
# -----------------------------
class MetricProjector(nn.Module):
    """
    Small MLP projector; output normalized embedding.
    """
    def __init__(self, dim: int, proj_dim: Optional[int] = None):
        super().__init__()
        proj_dim = proj_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, proj_dim),
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=1)


class EncProj(nn.Module):
    """
    Wrap encoder.encode + projector for embeddings used in SupCon and clustering.
    """
    def __init__(self, encoder, projector: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x):
        z = self.encoder.encode(x)          # [B, D]
        z = F.normalize(z, dim=1)
        zp = self.projector(z)              # [B, D'] normalized
        return zp


# -----------------------------
# Freeze logic
# -----------------------------
def freeze_encoder(model: EncProj, frozen: bool):
    for p in model.encoder.parameters():
        p.requires_grad = (not frozen)
    if frozen:
        model.encoder.eval()
    else:
        model.encoder.train()


# -----------------------------
# SupCon loss (weighted)
# -----------------------------
def supcon_loss_weighted(
    z: torch.Tensor,          # [B, D] normalized
    y: torch.Tensor,          # [B]
    w: torch.Tensor,          # [B] >=0
    temperature: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Supervised contrastive loss with per-anchor weighting.
    Anchor i contributes weight w_i.
    Positives are samples with same label (excluding self).
    """
    # similarity
    sim = (z @ z.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0]  # stability

    # positives mask
    y = y.view(-1, 1)
    pos = (y == y.T).float()
    pos.fill_diagonal_(0)

    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1, keepdim=True) + eps
    log_prob = sim - torch.log(denom)

    pos_count = pos.sum(dim=1) + eps
    loss_i = -(pos * log_prob).sum(dim=1) / pos_count

    # apply per-anchor weights (anchors high, pseudos low)
    w = w.clamp_min(0.0)
    loss = (loss_i * w).sum() / (w.sum() + eps)
    return loss


# -----------------------------
# Train loop (SupCon)
# -----------------------------
def train_supcon(
    model: EncProj,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int = 3,
    temperature: float = 0.2,
    reg_to_base: float = 0.0,
):
    """
    Train projector (and maybe encoder) with SupCon.
    Optional reg_to_base: keep zp close to base z (discourages geometry collapse).
    """
    model.train()
    for ep in range(epochs):
        total_loss, n = 0.0, 0

        for batch in loader:
            x = batch["x"].to(device).float()
            y = batch["label"].to(device).long()
            w = batch["weight"].to(device).float()

            # forward
            z_base = model.encoder.encode(x)
            z_base = F.normalize(z_base, dim=1)
            z = model.projector(z_base)

            loss = supcon_loss_weighted(z, y, w, temperature=temperature)

            if reg_to_base > 0:
                loss = loss + float(reg_to_base) * (z - z_base).pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        print(f"[train-supcon] ep {ep+1:02d} loss={total_loss/max(n,1):.4f}")


# -----------------------------
# Embedding extraction for clustering
# -----------------------------
@torch.no_grad()
def extract_embeddings_projected(model: EncProj, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    Z, Y = [], []
    for batch in loader:
        x = batch["x"].to(device).float()
        y = batch["label"].cpu().numpy()
        z = model(x).cpu().numpy()  # projected embedding
        Z.append(z)
        Y.append(y)
    return np.concatenate(Z, axis=0), np.concatenate(Y, axis=0)


@torch.no_grad()
def eval_clustering_projected(model: EncProj, test_loader, device, n_classes, k_multipliers=(1, 2, 3)):
    Z_test, y_test = extract_embeddings_projected(model, test_loader, device)
    _, _, _, all_results = evaluate_clustering(
        Z_test, y_test,
        base_k=n_classes,
        method="gmm",
        k_multipliers=k_multipliers,
    )
    return all_results


# -----------------------------
# "Prototype head eval" (no kmeans)
# -----------------------------
@torch.no_grad()
def eval_prototype_accuracy_nmi_ari(
    model: EncProj,
    labeled_idx: List[int],
    train_ds_plain: UrbanSoundDataset,
    test_loader: DataLoader,
    device: str,
    n_classes: int,
):
    """
    Build prototypes in projected space from anchors only.
    Predict each test sample by nearest prototype (cosine).
    Report acc/NMI/ARI.
    """
    model.eval()

    # compute anchor embeddings + labels
    anchors_x = []
    anchors_y = []
    for i in labeled_idx:
        item = train_ds_plain[int(i)]
        anchors_x.append(item["x"])
        anchors_y.append(int(item["label"]))

    anchors_x = torch.stack(anchors_x, dim=0).to(device).float()
    anchors_y = np.array(anchors_y, dtype=np.int64)

    z_anchor = model(anchors_x)  # [A, D] normalized
    z_anchor_np = z_anchor.detach().cpu().numpy()

    # prototypes
    protos = np.zeros((n_classes, z_anchor_np.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        m = anchors_y == c
        if m.sum() > 0:
            protos[c] = z_anchor_np[m].mean(axis=0)
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)

    # test
    y_true_all, y_pred_all = [], []
    for batch in test_loader:
        x = batch["x"].to(device).float()
        y = batch["label"].cpu().numpy()

        z = model(x).detach().cpu().numpy()
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

        sim = z @ protos.T
        pred = sim.argmax(axis=1).astype(np.int64)

        y_true_all.append(y)
        y_pred_all.append(pred)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    acc = accuracy_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    print(f"[proto eval] acc={acc:.3f}  NMI={nmi:.3f}  ARI={ari:.3f}")
    return {"acc": acc, "nmi": nmi, "ari": ari}


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_clips, test_clips, _ = get_train_test_clips_n_labels()
    label2id = create_label_map(train_clips + test_clips)
    n_classes = len(label2id)

    train_ds_plain = UrbanSoundDataset(
        clips=train_clips,
        label2id=label2id,
        precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
        transform=None,
        mode="ae",
        folds=[0,1,2,3,4,5,6,7,8,9],
    )
    test_ds = UrbanSoundDataset(
        clips=test_clips,
        label2id=label2id,
        precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
        transform=None,
        mode="ae",
        folds=[10],
    )

    train_loader_plain = DataLoader(train_ds_plain, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)

    # encoder
    encoder = ResNetEncoder(latent_dim=cfg.latent_dim, depth="layer2", pretrained=True).to(device)
    state = torch.load(r"C:\data\models\ae_resnet_181225.pth", map_location=device)
    encoder.load_state_dict(state)

    # projector model
    projector = MetricProjector(cfg.latent_dim, proj_dim=cfg.latent_dim).to(device)
    model = EncProj(encoder, projector).to(device)

    # y_train
    y_train = np.array([int(train_ds_plain[i]["label"]) for i in range(len(train_ds_plain))], dtype=np.int64)

    # anchors
    from weak_sup.get_labels import select_n_per_class
    labeled_idx = select_n_per_class(train_ds_plain, n_per_class=5, seed=0, min_conf=None, prefer_high_conf=True)
    labeled_set = set(int(i) for i in labeled_idx)
    print("Anchor count:", len(labeled_idx))

    # -------------------
    # Warm-start SupCon on anchors only
    # -------------------
    freeze_encoder(model, frozen=True)

    # optimizer: projector only
    opt = torch.optim.AdamW(model.projector.parameters(), lr=3e-4, weight_decay=1e-4)

    anchor_only_ds = PseudoLabelDataset(
        base_ds=train_ds_plain,
        labeled_idx=labeled_idx,
        pseudo_map={},           # none
        pseudo_conf=None,
        transform=None,          # start stable
        pseudo_w=0.0,
    )
    anchor_loader = DataLoader(anchor_only_ds, batch_size=min(cfg.batch_size, len(anchor_only_ds)),
                               shuffle=True, pin_memory=True)

    print("Warm-start SupCon on anchors...")
    train_supcon(model, anchor_loader, opt, device, epochs=30, temperature=0.2, reg_to_base=0.2)

    print("Clustering baseline after warm-start (projected embeddings):")
    base_res = eval_clustering_projected(model, test_loader, device, n_classes, k_multipliers=(1,2,3,4))
    for k, res in base_res.items():
        print(f"k={k:3d}  NMI={res['nmi']:.3f}  ARI={res['ari']:.3f}")

    eval_prototype_accuracy_nmi_ari(model, labeled_idx, train_ds_plain, test_loader, device, n_classes)

    # -------------------
    # Expansion rounds
    # -------------------
    pseudo_map: Dict[int, int] = {}
    pseudo_conf: Dict[int, float] = {}

    for round_id in range(1, 20):
        # keep encoder frozen most rounds
        if round_id <= 6:
            freeze_encoder(model, frozen=True)
            opt = torch.optim.AdamW(model.projector.parameters(), lr=5e-4, weight_decay=1e-4)
        else:
            freeze_encoder(model, frozen=False)
            opt = torch.optim.AdamW([
                {"params": model.projector.parameters(), "lr": 3e-4},
                {"params": model.encoder.parameters(), "lr": 2e-5},
            ], weight_decay=1e-5)

        # teacher embeddings: use ENCODER (not projector) for stability
        # Z_train, _ = extract_embeddings(model.encoder, train_loader_plain, device, normalize=True)
        Z_train, _ = extract_embeddings_projected(model, train_loader_plain, device)  # projected + normalized

        # pred, conf, margin = prototype_teacher(
        #     Z_all=Z_train,
        #     labeled_idx=labeled_idx,
        #     y_all=y_train,
        #     n_classes=n_classes,
        #     temp=0.10,
        #     reject_min_prob=0.35,
        #     reject_min_margin=0.02,
        # )
        pred, conf, margin, proto_id = subproto_teacher(
            Z_all=Z_train,  # you're using projected embeddings already
            labeled_idx=labeled_idx,
            y_all=y_train,
            n_classes=n_classes,
            pseudo_map=pseudo_map,
            pseudo_conf=pseudo_conf,
            max_total_subprotos=10,  # <<< your constraint
            min_per_class=1,
            max_per_class=6,
            temp=0.07,
            reject_min_prob=0.35,
            reject_min_margin=0.02,
            pseudo_conf_thr_for_protos=0.70,
            seed=round_id,  # small variation ok
        )

        pseudo_idx, per_class_counts = select_pseudo_indices_topN_per_class(
            pred, conf, labeled_set, n_classes,
            top_n=10,
            min_conf=0.65,
            fallback_min_conf=0.45,
            min_per_class=3,
        )

        # accumulate pseudo labels + confidence (persistent)
        new_added = 0
        for i in pseudo_idx:
            ii = int(i)
            if ii in pseudo_map:
                continue
            pseudo_map[ii] = int(pred[ii])
            pseudo_conf[ii] = float(conf[ii])
            new_added += 1

        rej = float((pred == -1).mean())
        w_p = pseudo_weight(round_id)
        print(f"\n[Round {round_id}] pseudo_total={len(pseudo_map)}  new={new_added}  reject={rej:.3f}  w_pseudo={w_p:.2f}")
        print("per-class new:", per_class_counts)

        mix_ds = PseudoLabelDataset(
            base_ds=train_ds_plain,
            labeled_idx=labeled_idx,
            pseudo_map=pseudo_map,
            pseudo_conf=pseudo_conf,
            transform=None if round_id <= 2 else StrongContrastiveAug(),  # add aug later
            pseudo_w=w_p,
            conf_power=1.0,  # try 2.0 if you want harsher down-weighting of mid-conf pseudos
        )
        mix_loader = DataLoader(mix_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)

        train_supcon(model, mix_loader, opt, device, epochs=5, temperature=0.2, reg_to_base=0.1)

        res = eval_clustering_projected(model, test_loader, device, n_classes, k_multipliers=(1,2,3,4))
        print("[Eval clustering projected]")
        for k, r in res.items():
            print(f"k={k:3d}  NMI={r['nmi']:.3f}  ARI={r['ari']:.3f}")

        eval_prototype_accuracy_nmi_ari(model, labeled_idx, train_ds_plain, test_loader, device, n_classes)


if __name__ == "__main__":
    main()
