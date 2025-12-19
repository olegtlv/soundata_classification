import torch
import numpy as np
import torch.nn.functional as F

def extract_embeddings_all(model, loader, device, normalize=True):
    Z, y, idx = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device).float()
            # if your dataset returns index, use it; otherwise add it (recommended)
            b_idx = batch.get("index", None)
            if b_idx is None:
                # fallback: can't track indices -> strongly recommend adding index in Dataset
                raise ValueError("Batch missing 'index'. Add it in Dataset return dict.")
            b_idx = b_idx.cpu().numpy()

            # support your encoders: either model.encode or model(x)->(recon,z)
            if hasattr(model, "encode"):
                z = model.encode(x)
            else:
                out = model(x)
                z = out[1] if isinstance(out, (tuple, list)) else out

            if normalize:
                z = F.normalize(z, dim=1)

            Z.append(z.cpu().numpy())
            y.append(batch["label"].cpu().numpy())
            idx.append(b_idx)

    return np.concatenate(Z), np.concatenate(y), np.concatenate(idx)
