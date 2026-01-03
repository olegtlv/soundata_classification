# training/contrastive_loss.py
import torch
import torch.nn.functional as F


def contrastive_loss(z1, z2, temperature=0.1, weights=None, normalize=True):
    """
    SimCLR NT-Xent loss.

    z1, z2: [B, D] embeddings (not necessarily normalized)
    temperature: scalar
    weights: optional [B] tensor with per-sample weights (e.g. confidence).

    Returns: scalar loss.
    """
    # Normalize embeddings
    if normalize:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

    B = z1.size(0)
    device = z1.device

    # Concatenate embeddings: [2B, D]
    z = torch.cat([z1, z2], dim=0)          # z[0..B-1] = z1, z[B..2B-1] = z2

    # Cosine similarity matrix: [2B, 2B]
    sim = torch.matmul(z, z.T)              # since z is normalized, this is cosine sim
    sim = sim / temperature

    # Mask out self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=device)
    sim = sim.masked_fill(mask, float("-inf"))

    # Positive index for each sample:
    # for i in [0..B-1], positive is i+B
    # for i in [B..2B-1], positive is i-B
    pos_indices = torch.arange(2 * B, device=device)
    pos_indices = (pos_indices + B) % (2 * B)

    # Per-sample loss (no reduction yet): [2B]
    loss_vec = F.cross_entropy(sim, pos_indices, reduction="none")

    if weights is not None:
        # weights: [B] → make [2B] by repeating for both views
        w = weights.view(-1).to(device)     # [B]
        if w.numel() != B:
            raise ValueError(f"weights must be length B={B}, got {w.numel()}")
        w = torch.cat([w, w], dim=0)        # [2B]

        # Weighted mean: sum(w * loss) / sum(w)
        loss = (loss_vec * w).sum() / w.sum()
    else:
        loss = loss_vec.mean()

    return loss
