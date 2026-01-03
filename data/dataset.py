# data/dataset.py
from __future__ import annotations

from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn.functional as F
import random

from data.preprocess import preprocess_audio, to_logmel


class UrbanSoundDataset(Dataset):
    """
    Supports:
      - precomputed_file with X/y/(optional folds/confidence/salience)
      - on-the-fly clips + label2id

    Modes:
      - "ae" / "vae": returns {"x","x_clean",...}
      - "contrastive": returns {"x1","x2","x_clean",...} and optionally {"x_bg"} if requested
      - "supervised": returns {"x",...}

    Quality filtering:
      - use_quality_filter (default True) + thresholds min_confidence/min_salience
      - implemented via index_map so __len__/__getitem__ reflect the filtered set
    """

    def __init__(
        self,
        clips=None,
        label2id=None,
        precomputed_file: str | None = None,
        transform=None,
        mode: str = "ae",
        folds=None,
        # --- new flags ---
        use_quality_filter: bool = True,        # defaulted to True as you asked
        min_confidence: float = 1.0,
        req_salience: float = 1.0,
        # background mixing support (optional)
        return_bg: bool = False,
    ):
        self.data = clips
        self.label2id = label2id
        self.transform = transform
        self.mode = mode

        self.use_quality_filter = use_quality_filter
        self.min_confidence = min_confidence
        self.req_salience = req_salience
        self.return_bg = return_bg
        self.index_map = None  # <--- add


        if precomputed_file is not None:
            data = torch.load(precomputed_file)

            self.X = data["X"]  # [N,1,mel,time]
            self.y = data["y"]
            self.folds_tensor = data.get("folds", None)
            self.label2id = data.get("label2id", self.label2id)

            # OPTIONAL: per-sample confidence/salience if stored
            self.confidence = data.get("confidence", None)
            self.salience = data.get("salience", None)

            N = len(self.X)

            # force to 1D [N] float tensors to prevent broadcasting bugs
            if self.confidence is not None:
                self.confidence = torch.as_tensor(self.confidence).float().view(-1)
                if len(self.confidence) != N:
                    raise ValueError(f"confidence length {len(self.confidence)} != N {N}")
            if self.salience is not None:
                self.salience = torch.as_tensor(self.salience).float().view(-1)
                if len(self.salience) != N:
                    raise ValueError(f"salience length {len(self.salience)} != N {N}")

            # build keep mask
            keep = torch.ones(N, dtype=torch.bool)

            # fold filter
            if folds is not None and self.folds_tensor is not None:
                keep &= torch.isin(self.folds_tensor, torch.as_tensor(folds))

            # quality filter (default ON)
            if self.use_quality_filter:
                if self.confidence is not None and self.min_confidence is not None:
                    keep &= (self.confidence >= float(self.min_confidence))
                if self.salience is not None and self.req_salience is not None:
                    keep &= (self.salience == float(self.req_salience))

            self.index_map = torch.nonzero(keep, as_tuple=False).squeeze(1)

            # dynamic keep mask (over ORIGINAL N indices)
            self._dynamic_keep = None  # None means "no dynamic filter"
            self._dynamic_weights = None  # optional per-sample weights over original indices

            self.precomputed = True

        elif clips is not None and label2id is not None:
            self.clips = clips
            self.label2id = label2id
            self.precomputed = False
        else:
            raise ValueError("Either clips+label2id or precomputed_file must be provided")

    # ---- helpers ----
    def _pad_to_128_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, F, T]
        if x.shape[-1] == 126:
            x = F.pad(x, (1, 1))
        return x

    def _resolve_real_idx(self, idx: int) -> int:
        # map filtered idx -> original idx
        if getattr(self, "precomputed", False):
            return int(self.index_map[idx])
        return idx

    def _load_item_as_x(self, idx: int) -> torch.Tensor:
        """
        For on-the-fly mode: load clip idx and return [1,F,T] float tensor.
        """
        clip = self.clips[idx]
        y, sr, _, _ = preprocess_audio(clip)
        logmel = to_logmel(y, sr)
        x = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0)
        return x

    def _sample_bg(self, avoid_idx: int | None = None) -> torch.Tensor:
        """
        Samples background from the *same* filtered pool when precomputed,
        otherwise from clips list.
        """
        if getattr(self, "precomputed", False):
            n = len(self.index_map)
            if n <= 1:
                return self._pad_to_128_if_needed(self.X[self._resolve_real_idx(0)].clone())

            # try a couple times to avoid picking the same sample
            for _ in range(5):
                j_local = torch.randint(0, n, (1,)).item()
                j = int(self.index_map[j_local])
                if avoid_idx is None or j != avoid_idx:
                    x_bg = self.X[j].clone()
                    return self._pad_to_128_if_needed(x_bg)

            # fallback
            j_local = torch.randint(0, n, (1,)).item()
            j = int(self.index_map[j_local])
            x_bg = self.X[j].clone()
            return self._pad_to_128_if_needed(x_bg)

        # on-the-fly
        n = len(self.clips)
        if n <= 1:
            return self._pad_to_128_if_needed(self._load_item_as_x(0))

        for _ in range(5):
            j = random.randrange(n)
            if avoid_idx is None or j != avoid_idx:
                return self._pad_to_128_if_needed(self._load_item_as_x(j))

        return self._pad_to_128_if_needed(self._load_item_as_x(random.randrange(n)))


    def set_index_map(self, index_map):
        """
        index_map: 1D iterable of *real indices* into self.X (precomputed)
        """
        if index_map is None:
            self.index_map = None
            return
        if isinstance(index_map, torch.Tensor):
            self.index_map = index_map.long()
        else:
            self.index_map = torch.tensor(list(index_map), dtype=torch.long)

    def __len__(self):
        if getattr(self, "precomputed", False):
            if self.index_map is not None:
                return int(self.index_map.numel())
            return len(self.X)
        else:
            # optional: for on-the-fly you can also support index_map similarly
            return len(self.clips)


    # ---- main ----
    def __getitem__(self, idx: int):
        # ------------------------
        # Precomputed mode
        # ------------------------
        if getattr(self, "precomputed", False):
            real_idx = int(self.index_map[idx]) if self.index_map is not None else idx

            x = self.X[real_idx].clone()
            label_id = self.y[real_idx]

            # (keep your existing metadata logic)
            # per-sample metadata if exists, else defaults
            if getattr(self, "salience", None) is not None:
                salience = self.salience[real_idx].float()
            else:
                salience = torch.tensor(1.0, dtype=torch.float32)

            if getattr(self, "confidence", None) is not None:
                confidence = self.confidence[real_idx].float()
            else:
                confidence = torch.tensor(1.0, dtype=torch.float32)

            if x.shape[-1] == 126:
                x = F.pad(x, (1, 1))
        else:
            real_idx = idx
            clip = self.clips[idx]
            x = self._load_item_as_x(idx)
            x = self._pad_to_128_if_needed(x)

            label_id = self.label2id[clip.class_label]
            salience = torch.tensor(getattr(clip, "salience", 1.0), dtype=torch.float32)
            confidence = torch.tensor(getattr(getattr(clip, "tags", None), "confidence", 1.0), dtype=torch.float32)

        # ------------------------------------------------
        # AE / VAE
        # ------------------------------------------------
        if self.mode in ("ae", "vae"):
            x_clean = x.clone()
            x_aug = self.transform(x) if self.transform else x
            return {
                "x": x_aug,
                "x_clean": x_clean,
                "label": label_id,
                "salience": salience,
                "confidence": confidence,
                "index": real_idx,   # global index (useful for alignment/debug)
            }

        # ------------------------------------------------
        # Contrastive (SimCLR / BYOL)
        # ------------------------------------------------
        elif self.mode == "contrastive":
            # sample background from a different random index (optional)
            if getattr(self, "precomputed", False):
                j = torch.randint(0, len(self.X), (1,)).item()
                x_bg = self.X[j].clone()
                if x_bg.shape[-1] == 126:
                    x_bg = F.pad(x_bg, (1, 1))
            else:
                x_bg = None  # keep simple for now in on-the-fly mode

            # If transform supports (x, x_bg) and returns (x1, x2, sal_w)
            if self.transform is None:
                raise ValueError("contrastive mode requires a transform (e.g., TwoViewAug).")

            out = None
            try:
                out = self.transform(x, x_bg=x_bg)  # new API
            except TypeError:
                out = self.transform(x)  # fallback

            # Parse outputs
            if isinstance(out, (tuple, list)) and len(out) == 3:
                x1, x2, sal_w = out
            elif isinstance(out, (tuple, list)) and len(out) == 2:
                x1, x2 = out
                sal_w = torch.tensor(1.0)
            else:
                # fallback: single-view transform called twice
                x1 = out
                x2 = self.transform(x)
                sal_w = torch.tensor(1.0)

            return {
                "x_clean": x.clone(),
                "x_bg": x_bg if x_bg is not None else torch.zeros_like(x),
                "x1": x1,
                "x2": x2,
                "sal_w": sal_w,  # <-- NEW (scalar)
                "label": label_id,
                "salience": salience,  # your existing metadata if you have it
                "confidence": confidence,
                "index": idx,
            }

        # ------------------------------------------------
        # Supervised classifier
        # ------------------------------------------------
        if self.mode == "supervised":
            x_out = self.transform(x) if self.transform else x
            return {
                "x": x_out,
                "label": label_id,
                "salience": salience,
                "confidence": confidence,
                "index": real_idx,
            }

        raise ValueError(f"Unknown mode: {self.mode}")

    def set_dynamic_keep(self, keep_mask: torch.Tensor | None):
        """
        keep_mask: bool tensor shape [N_original] over self.X indices.
        If None -> disable dynamic filtering.
        """
        if keep_mask is None:
            self._dynamic_keep = None
            return
        keep_mask = torch.as_tensor(keep_mask).bool().view(-1)
        if keep_mask.numel() != len(self.X):
            raise ValueError(f"keep_mask has {keep_mask.numel()} elems, expected {len(self.X)}")
        self._dynamic_keep = keep_mask

    def set_dynamic_weights(self, weights: torch.Tensor | None):
        """
        weights: float tensor shape [N_original] over self.X indices.
        """
        if weights is None:
            self._dynamic_weights = None
            return
        weights = torch.as_tensor(weights).float().view(-1)
        if weights.numel() != len(self.X):
            raise ValueError(f"weights has {weights.numel()} elems, expected {len(self.X)}")
        self._dynamic_weights = weights


class PseudoLabeledDataset(Dataset):
    def __init__(self, base_ds: Dataset, pseudo_labels):
        assert len(base_ds) == len(pseudo_labels)
        self.base_ds = base_ds
        self.pseudo_labels = np.array(pseudo_labels, dtype=np.int64)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]

        # if base is UrbanSoundDataset in AE mode it returns dict with "x"
        if isinstance(sample, dict):
            x = sample.get("x", sample.get("x_clean"))
        else:
            # fallback if someone wrapped/changed base ds
            x = sample[0]

        pseudo = int(self.pseudo_labels[idx])
        return x, pseudo
