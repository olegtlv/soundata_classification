from torch import nn
from config import Config
import torch.nn.functional as F


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
                 proj_dim=128, hidden_dim=256):
        super().__init__()
        self.encoder = ae_backbone

        # SimCLR-style projector: z -> p
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

    def forward(self, x):
        # ConvAE encode: [B,1,F,T] -> [B,latent_dim]
        z = self.encoder.encode(x)          # [B, latent_dim]
        z = F.normalize(z, dim=1)

        p = self.projector(z)               # [B, proj_dim]
        p = F.normalize(p, dim=1)

        return p
