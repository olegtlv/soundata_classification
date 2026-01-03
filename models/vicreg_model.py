# models/vicreg_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, bn: bool = True):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            if bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VICRegModel(nn.Module):
    """
    Wraps an encoder and adds a projector used only for the VICReg objective.
    You cluster on 'z' (encoder features), not on the projection.
    """
    def __init__(
        self,
        encoder: nn.Module,
        encoder_out_dim: int,
        proj_hidden_dim: int = 512,
        proj_out_dim: int = 256,
        proj_layers: int = 2,
        proj_bn: bool = True,
        z_norm: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = MLP(
            in_dim=encoder_out_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim,
            num_layers=proj_layers,
            bn=proj_bn,
        )
        self.z_norm = z_norm

    def forward(self, x):
        """
        Returns:
          z: encoder features (use these for k-means)
          y: projected features (used for VICReg loss)
        """
        z = self._encode(x)
        if self.z_norm:
            z = F.normalize(z, dim=1)
        y = self.projector(z)
        return z, y

    def _encode(self, x):
        """
        Works with:
          - your ResNetEncoder: returns [B, D]
          - your ConvAutoencoder: returns [B, D] if we call encode(x)
        """
        # If your encoder has 'encode' (ConvAutoencoder), use it:
        if hasattr(self.encoder, "encode"):
            z = self.encoder.encode(x)
        else:
            z = self.encoder(x)

        # Flatten safely: [B, ...] -> [B, D]
        if z.dim() > 2:
            z = z.flatten(start_dim=1)
        return z
