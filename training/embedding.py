# training/embedding.py
import torch
import torch.nn.functional as F


def extract_embeddings(model, loader, device, normalize=False):
    model.eval()
    Z, labels = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x_clean"].to(device).float()
            label = batch["label"]

            z = model.encode(x)
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