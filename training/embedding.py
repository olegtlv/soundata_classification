# training/embedding.py
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np


def extract_embeddings(model, loader, device, normalize=False):
    model.eval()
    Z, labels = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x_clean"].to(device).float()
            label = batch["label"]

            z = model.encode_only(x)
            if normalize:
                z = F.normalize(z, dim=1)

            Z.append(z.cpu())
            labels.append(label.cpu())

    return torch.cat(Z), torch.cat(labels)


def extract_embeddings_VAE(model, loader, device):
    model.eval()
    Z, labels = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x_clean"].to(device).float()
            label = batch["label"]

            z = model.encode(x)[0]
            z = F.normalize(z, dim=1)

            Z.append(z.cpu())
            labels.append(label.cpu())

    return torch.cat(Z), torch.cat(labels)

def pca_transform(Z, n_components=64, whiten=True, l2_after=True, random_state=42):
    Z = np.asarray(Z, dtype=np.float32)

    pca = PCA(n_components=n_components, svd_solver="auto", random_state=random_state)
    Zp = pca.fit_transform(Z).astype(np.float32)   # [N, n_components]

    if whiten:
        # sklearn's PCA has a whiten=True option, but it rescales differently depending on solver;
        # doing it explicitly is clear and stable:
        Zp = Zp / (np.sqrt(pca.explained_variance_.astype(np.float32)) + 1e-8)

    if l2_after:
        Zp = Zp / (np.linalg.norm(Zp, axis=1, keepdims=True) + 1e-8)

    return Zp, pca.explained_variance_ratio_
