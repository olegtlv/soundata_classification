from torch import nn
from config import Config
import torch.nn.functional as F
import torch

class Prototypes(nn.Module):
    def __init__(self, in_dim, num_prototypes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_prototypes, in_dim) * 0.01)

    def forward(self, x):  # x: [B,D]
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return x @ w.t()  # logits: [B, K]


# class SimCLRModel(nn.Module):
#     def __init__(self, ae_backbone, latent_dim=128):
#         super().__init__()
#         self.encoder = ae_backbone      # ConvAutoencoder
#         self.projector = nn.Sequential(
#             nn.Linear(Config.latent_dim, latent_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(latent_dim, latent_dim),
#         )
#
#     def forward(self, x):
#         z = self.encoder.encode_only(x)   # [B, latent_dim]
#         p = self.projector(z)             # [B, proj_dim]
#         p = F.normalize(p, dim=1)
#         return p


class SimCLRModel(nn.Module):
    """
    SimCLR wrapper:
    - encoder: any module with .encode(x) -> [B, latent_dim]
    - projector: 3-layer MLP head (SimCLR-style)
    """
    def __init__(self, ae_backbone, latent_dim=128,
                 proj_dim=128, hidden_dim=256, normalize_latent=False,
                 num_prototypes=256):
        super().__init__()
        self.encoder = ae_backbone
        self.normalize_latent = normalize_latent
        self.prototypes = Prototypes(in_dim=proj_dim, num_prototypes=num_prototypes)

        # SimCLR-style projector: z -> p
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        # ConvAE encode: [B,1,F,T] -> [B,latent_dim]
        z = self.encoder.encode(x)          # [B, latent_dim]
        z = F.normalize(z, dim=1)

        p = self.projector(z)               # [B, proj_dim]
        p = F.normalize(p, dim=1)

        return p

    def encode_only(self, x):
        z = self.encoder.encode(x)
        return F.normalize(z, dim=1) if self.normalize_latent else z
