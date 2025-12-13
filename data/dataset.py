# data/dataset.py
from torch.utils.data import Dataset
import torch
from data.preprocess import preprocess_audio, to_logmel
import numpy as np
import torch.nn.functional as F

class UrbanSoundDataset(Dataset):
    def __init__(self, clips=None, label2id=None,
                 precomputed_file=None,
                 transform=None,
                 mode="ae", folds=None):
        self.data = clips
        self.label2id = label2id
        self.transform = transform
        self.mode = mode

        if precomputed_file is not None:
            data = torch.load(precomputed_file)
            self.X = data["X"]   # [N,1,mel,time]
            self.y = data["y"]
            self.folds_tensor = data.get("folds", None)
            self.label2id = data["label2id"]

            # filter folds if requested
            if folds is not None and self.folds_tensor is not None:
                mask = torch.isin(self.folds_tensor, torch.tensor(folds))
                self.X = self.X[mask]
                self.y = self.y[mask]

            self.precomputed = True
        elif clips is not None and label2id is not None:
            self.clips = clips
            self.label2id = label2id
            self.precomputed = False
        else:
            raise ValueError("Either clips+label2id or precomputed_file must be provided")

    def __len__(self):
        return len(self.X) if getattr(self, "precomputed", False) else len(self.clips)

    def __getitem__(self, idx):
        # ------------------------
        # Precomputed mode
        # ------------------------
        if getattr(self, "precomputed", False):
            x = self.X[idx].clone()
            label_id = self.y[idx]
            salience = getattr(self, "salience", torch.ones_like(label_id, dtype=torch.float32))
            confidence = getattr(self, "confidence", torch.ones_like(label_id, dtype=torch.float32))
            if x.shape[-1] == 126:
                x = F.pad(x, (1, 1))  # pad time dimension to 128
        # ------------------------
        # On-the-fly mode
        # ------------------------
        else:
            clip = self.clips[idx]
            y, sr, _, _ = preprocess_audio(clip)
            logmel = to_logmel(y, sr)
            x = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0)
            label_id = self.label2id[clip.class_label]
            # Get clip attributes if they exist
            salience = torch.tensor(getattr(clip, "salience", 1.0), dtype=torch.float32)
            confidence = torch.tensor(getattr(clip.tags, "confidence", 1.0), dtype=torch.float32)

        # ------------------------------------------------
        # AE mode: returns x_aug + x_clean
        # ------------------------------------------------
        if (self.mode == "ae") | (self.mode == "vae"):
            x_clean = x.clone()
            x_aug = self.transform(x) if self.transform else x

            return {
                "x": x_aug,
                "x_clean": x_clean,
                "label": label_id,
                "salience": salience,
                "confidence": confidence
            }

        # ------------------------------------------------
        # Contrastive (SimCLR / BYOL)
        # ------------------------------------------------
        elif self.mode == "contrastive":
            x1 = self.transform(x)
            x2 = self.transform(x)
            x_clean = x.clone()
            return {
                "x_clean": x_clean,
                "x1": x1,
                "x2": x2,
                "label": label_id,
                "salience": salience,
                "confidence": confidence
            }
        # ------------------------------------------------
        # Supervised classifier
        # ------------------------------------------------
        elif self.mode == "supervised":
            if self.transform:
                x = self.transform(x)
            return {
                "x": x,
                "label": label_id,
                "salience": salience,
                "confidence": confidence
            }
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class PseudoLabeledDataset(Dataset):
    def __init__(self, base_ds, pseudo_labels):
        assert len(base_ds) == len(pseudo_labels)
        self.base_ds = base_ds
        self.pseudo_labels = np.array(pseudo_labels, dtype=np.int64)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        x = sample["x"]       # from UrbanSoundDataset
        # salience/confidence/real label are ignored for DeepCluster-lite
        pseudo = int(self.pseudo_labels[idx])
        return x, pseudo