# training/embedding.py
import torch

def extract_embeddings(model, loader, device):
    model.eval()
    Z, labels = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x_clean"].to(device).float()
            label = batch["label"]

            z = model.encode(x)

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

            Z.append(z.cpu())
            labels.append(label.cpu())

    return torch.cat(Z), torch.cat(labels)