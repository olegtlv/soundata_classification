# precompute_logmels.py
import torch
from data.data_access import load_dataset, create_label_map
from data.preprocess import preprocess_audio, to_logmel
from tqdm import tqdm
CACHE_DIR = r"C:\data\soundata_processed\urbansound8k_logmels.pt"

# Load clips
clips = load_dataset("urbansound8k")
label2id = create_label_map(clips)

logmels = []
labels = []
folds = []
saliences = []
confidences = []


for clip in tqdm(clips, desc="Processing clips"):
    y, sr, _, fold = preprocess_audio(clip)
    logmel = to_logmel(y, sr)          # shape [mel, time]
    logmels.append(torch.tensor(logmel, dtype=torch.float32))
    labels.append(label2id[clip.class_label])
    folds.append(fold)
    saliences.append(torch.tensor(getattr(clip, "salience", 1.0), dtype=torch.float32))
    confidences.append(torch.tensor(getattr(getattr(clip, "tags", {}), "confidence", 1.0), dtype=torch.float32))

# Stack tensors
X = torch.stack([l.unsqueeze(0) for l in logmels])  # [N, 1, mel, time]
y = torch.tensor(labels, dtype=torch.long)
folds = torch.tensor(folds, dtype=torch.long)
saliences = torch.stack(saliences)     # [N]
confidences = torch.stack(confidences) # [N]

# Save single file
torch.save({
    "X": X,
    "y": y,
    "folds": folds,
    "salience": saliences,
    "confidence": confidences,
    "label2id": label2id
}, CACHE_DIR)
print("Saved precomputed logmels!")