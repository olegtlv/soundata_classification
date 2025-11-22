import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(64 * 8 * 16, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 16)
        self.dec1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.dbn1 = nn.BatchNorm2d(32)
        self.dbn2 = nn.BatchNorm2d(16)

    def encode(self, x):
        x = F.relu(self.bn1(self.enc1(x)))
        x = F.relu(self.bn2(self.enc2(x)))
        x = F.relu(self.bn3(self.enc3(x)))
        x = self.flatten(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z, target_shape=None):
        x = F.relu(self.fc_dec(z))
        x = x.view(-1, 64, 8, 16)
        x = F.relu(self.dbn1(self.dec1(x)))
        x = F.relu(self.dbn2(self.dec2(x)))
        x = torch.sigmoid(self.dec3(x))
        if target_shape is not None:
            # resize to match input
            x = F.interpolate(x, size=target_shape[-2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        z = self.encode(x)
        z = F.normalize(z, dim=1)
        recon = self.decode(z, target_shape=x.shape)
        return recon, z

    def encode_only(self, x):
        z = self.encode(x)
        return F.normalize(z, dim=1)
