import torch.nn as nn
import torch.optim as optim
import torch

class DeepClusterHead(nn.Module):
    def __init__(self, encoder, latent_dim, n_clusters):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(latent_dim, n_clusters)

    def forward(self, x):
        # with torch.no_grad():          # encoder frozen
        _, z = self.encoder(x)     # (recon, z)
        logits = self.head(z)
        return logits
