from torch import nn
from config import Config
import torch.nn.functional as F


class SimCLRModel(nn.Module):
    def __init__(self, ae_backbone, latent_dim=128):
        super().__init__()
        self.encoder = ae_backbone      # ConvAutoencoder
        self.projector = nn.Sequential(
            nn.Linear(Config.latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        z = self.encoder.encode_only(x)   # [B, latent_dim]
        p = self.projector(z)             # [B, proj_dim]
        p = F.normalize(p, dim=1)
        return p
