# main.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
from config import Config
from data.data_access import get_train_test_clips_n_labels, create_label_map
from data.dataset import UrbanSoundDataset
from data.augmentations import get_ae_augment, ContrastiveAug, SafeContrastiveAug, StrongContrastiveAug
from models.convAE_model import ConvAutoencoder
from models.resnetEncoder import ResNetEncoder
from training.clustering_eval import evaluate_clustering
from training.embedding import extract_embeddings

from weak_sup.extract_embeddings import extract_embeddings_all
from weak_sup.knn_predict import knn_predict
from weak_sup.prototype import anchors_from_arrays, compute_prototypes, prototype_predict

# ======================================================
# Load config
# ======================================================
cfg = Config()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# Dataset
# ======================================================
train_clips, test_clips, labels = get_train_test_clips_n_labels()
label2id = create_label_map(train_clips + test_clips)

aug = get_ae_augment() if cfg.model_type in ['ae', 'vae'] else StrongContrastiveAug()

train_ds = UrbanSoundDataset(
    clips=train_clips,
    label2id=label2id,
    precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
    transform=aug,
    mode='contrastive',  # cfg.model_type,
    folds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)

test_ds = UrbanSoundDataset(
    clips=test_clips,
    label2id=label2id,
    precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
    transform=None,
    mode='ae',
    folds=[10]
)

train_loader = DataLoader(train_ds,
                         batch_size=cfg.batch_size,
                         shuffle=True,
                         pin_memory=True,
                          )
test_loader = DataLoader(test_ds,
                         batch_size=cfg.batch_size,
                         shuffle=False,
                         pin_memory=True,
                         )
encoder = ResNetEncoder(latent_dim=cfg.latent_dim, depth="layer2", pretrained=True).to(device)
state = torch.load(r"C:\data\models\ae_resnet_181225.pth", map_location=device)
encoder.load_state_dict(state)

# encoder = ConvAutoencoder(latent_dim=cfg.latent_dim).to(device)
# state = torch.load(r"C:\data\models\ae_byol_181225.pth", map_location=device)
# encoder.load_state_dict(state)
device = next(encoder.parameters()).device

Z, labels = extract_embeddings(encoder, test_loader, device)
# Z, labels = extract_embeddings(model.encoder, test_loader, device, normalize=True)
# Z, labels = extract_embeddings(model.online_encoder, test_loader, device, normalize=True)
nmi, ari, preds, all_results = evaluate_clustering(Z,
                                                   labels,
                                                   base_k=len(label2id),
                                                   method="kmeans",
                                                   k_multipliers=(1, 2, 3)
                                                   )
for k, res in all_results.items():
    print(f"k={k:3d}  NMI={res['nmi']:.3f}  ARI={res['ari']:.3f}")

Z, y_true, idx_all = extract_embeddings_all(encoder, test_loader, device, normalize=True)
print(Z.shape, y_true.shape, idx_all.shape)

anchors = anchors_from_arrays(idx_all, y_true, n_per_class=5)
print({c: len(v) for c, v in anchors.items()})

protos = compute_prototypes(Z, idx_all, anchors)
pred, margin, sim = prototype_predict(Z, protos, reject_margin=0.05)

reject_rate = (pred == -1).mean()
print("reject_rate:", reject_rate)


mask = pred != -1
print("NMI:", normalized_mutual_info_score(y_true[mask], pred[mask]))
print("ARI:", adjusted_rand_score(y_true[mask], pred[mask]))

pred, conf = knn_predict(Z, idx_all, anchors)
mask = np.ones_like(pred)
print("NMI:", normalized_mutual_info_score(y_true, pred))
print("ARI:", adjusted_rand_score(y_true, pred))
print()