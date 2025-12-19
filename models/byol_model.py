import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(x, eps=1e-8):
    return x / (x.norm(dim=1, keepdim=True) + eps)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOLModel(nn.Module):
    """
    BYOL = online encoder+proj+pred trained to match target encoder+proj (EMA).
    Expects an encoder object with .encode(x) -> [B, latent_dim].
    """
    def __init__(
        self,
        encoder,
        latent_dim: int,
        proj_dim: int = 256,
        proj_hidden: int = 512,
        pred_hidden: int = 512,
        ema_m: float = 0.996,
    ):
        super().__init__()
        self.online_encoder = encoder
        self.online_projector = MLP(latent_dim, proj_hidden, proj_dim)
        self.online_predictor = MLP(proj_dim, pred_hidden, proj_dim)

        # target is a deep copy (same architecture), updated by EMA only
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.ema_m = ema_m

    @torch.no_grad()
    def update_target(self, m: float = None):
        """EMA update of target network."""
        if m is None:
            m = self.ema_m

        for p_o, p_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(m).add_((1.0 - m) * p_o.data)

        for p_o, p_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            p_t.data.mul_(m).add_((1.0 - m) * p_o.data)

    def forward_online(self, x):
        z = self.online_encoder.encode(x)              # [B, latent_dim]
        p = self.online_projector(z)                   # [B, proj_dim]
        q = self.online_predictor(p)                   # [B, proj_dim]
        return z, p, q

    @torch.no_grad()
    def forward_target(self, x):
        z = self.target_encoder.encode(x)
        p = self.target_projector(z)
        return z, p


def byol_loss(q, p_target):
    """
    BYOL loss: 2 - 2 * cosine_similarity(normalized(q), normalized(p_target))
    """
    q = _l2_normalize(q)
    p_target = _l2_normalize(p_target)
    return 2.0 - 2.0 * (q * p_target).sum(dim=1).mean()
