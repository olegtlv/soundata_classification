# training/vicreg_loss.py
import torch
import torch.nn.functional as F


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    # x: [D, D]
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    y1: torch.Tensor,
    y2: torch.Tensor,
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
):
    """
    VICReg loss on projected reps y1,y2 : [B, Dp]
      - invariance: MSE(y1, y2)
      - variance: penalize std < 1
      - covariance: penalize off-diagonal covariance
    """
    # invariance
    sim = F.mse_loss(y1, y2)

    # variance
    y1 = y1 - y1.mean(dim=0)
    y2 = y2 - y2.mean(dim=0)

    std_y1 = torch.sqrt(y1.var(dim=0, unbiased=False) + eps)
    std_y2 = torch.sqrt(y2.var(dim=0, unbiased=False) + eps)
    var = (F.relu(1.0 - std_y1).mean() + F.relu(1.0 - std_y2).mean())

    # covariance
    b = y1.shape[0]
    cov_y1 = (y1.T @ y1) / (b - 1)
    cov_y2 = (y2.T @ y2) / (b - 1)
    cov = (off_diagonal(cov_y1).pow(2).mean() + off_diagonal(cov_y2).pow(2).mean())

    loss = sim_coeff * sim + var_coeff * var + cov_coeff * cov
    stats = {
        "loss": loss.detach(),
        "sim": sim.detach(),
        "var": var.detach(),
        "cov": cov.detach(),
    }
    return loss, stats
