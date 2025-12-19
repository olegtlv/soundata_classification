# deepcluster_lite.py
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import torch.nn as nn
from config import Config
from data.data_access import get_train_test_clips_n_labels, create_label_map
from data.dataset import UrbanSoundDataset, PseudoLabeledDataset
from data.augmentations import get_ae_augment
from models.convAE_model import ConvAutoencoder
from models.resnetEncoder import ResNetEncoder
from training.embedding import extract_embeddings
from training.clustering_eval import evaluate_clustering
from models.deep_cluster_lite import DeepClusterHead
from training.train_deep_cluster_head import train_deep_cluster_head

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 1. Data
# ============================
train_clips, test_clips, _ = get_train_test_clips_n_labels()
label2id = create_label_map(train_clips + test_clips)
n_clusters = cfg.n_clusters  #len(label2id)  # one cluster per class

ae_transform = None #get_ae_augment()  # same as in main

train_ds = UrbanSoundDataset(
    clips=train_clips,
    label2id=label2id,
    precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
    transform=ae_transform,
    mode="ae",
    folds=[1, 2, 3, 4, 5, 6, 7, 8, 9]
)

test_ds = UrbanSoundDataset(
    clips=test_clips,
    label2id=label2id,
    precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
    transform=None,
    mode="ae",
    folds=[10]
)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

# ============================
# 2. Load pretrained AE encoder
# ============================
# ae = ConvAutoencoder(latent_dim=cfg.latent_dim).to(device)
# ae = ResNetEncoder(latent_dim=cfg.latent_dim).to(device)
res_encoder = ResNetEncoder(latent_dim=cfg.latent_dim, depth="layer2", pretrained=True).to(device)

# state = torch.load(r"C:\data\models\audio_ae_sim_tuned_291125.pth", map_location=device)
state = torch.load(r"C:\data\models\ae_resnet_181225.pth", map_location=device)
res_encoder.load_state_dict(state)
# ---- FREEZE early encoder layers ----
def freeze(module):
    for p in module.parameters():
        p.requires_grad = False
def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True
# freeze(ae.enc1)
# freeze(ae.enc2)

encoder = res_encoder  # we'll keep updating this reference
encoder.eval()

# Baseline AE performance (before DeepCluster)
with torch.no_grad():
    Z_test_ae, y_test_ae = extract_embeddings(encoder, test_loader, device)
base_nmi, base_ari, preds, all_results  = evaluate_clustering(Z_test_ae, y_test_ae, base_k=cfg.n_clusters, k_multipliers=(1, 2, 3))
print(f"[Baseline AE] NMI = {base_nmi:.4f}, ARI = {base_ari:.4f}")

# ============================
# 3. DeepCluster iterations
# ============================
criterion = nn.CrossEntropyLoss()
num_iters = 5          # number of DeepCluster iterations
epochs_per_iter = 5   # epochs per iteration

for it in range(num_iters):
    print(f"\n=== DeepCluster iteration {it + 1}/{num_iters} ===")

    # 3a. Extract embeddings with current encoder
    encoder.eval()
    if it == 0:
        freeze(encoder)
    else:
        unfreeze(encoder)

    with torch.no_grad():
        Z_train, _ = extract_embeddings(encoder, train_loader, device)

    # 3b. K-means on train embeddings -> pseudo-labels
    f = 4
    kmeans = KMeans(
        n_clusters=n_clusters*f,
        n_init=20,
        random_state=0
    )
    cluster_ids_train = kmeans.fit_predict(Z_train)

    # 3c. Pseudo-labeled dataset / loader
    dc_train_ds = PseudoLabeledDataset(train_ds, cluster_ids_train)
    dc_train_loader = DataLoader(dc_train_ds, batch_size=cfg.batch_size, shuffle=True)
    # freeze(encoder.enc1)
    # freeze(encoder.enc2)
    # 3d. New DeepCluster head on top of current encoder
    dc_model =  DeepClusterHead(encoder, cfg.latent_dim, n_clusters*f).to(device)
    # reinit head to avoid label-permutation issues
    dc_model.head.reset_parameters()

    # different LRs: slow for encoder, fast for head
    optimizer = torch.optim.Adam([
        {"params": dc_model.encoder.parameters(), "lr": 1e-5},
        {"params": dc_model.head.parameters(), "lr": 1e-3},
    ])

    # 3e. Train encoder + head on pseudo-labels
    train_deep_cluster_head(
        dc_model, dc_train_loader, optimizer, criterion, device,
        epochs=epochs_per_iter
    )

    # update encoder reference with fine-tuned weights
    encoder = dc_model.encoder

    # (optional) quick eval after each iteration
    encoder.eval()
    with torch.no_grad():
        Z_test_iter, y_test_iter = extract_embeddings(encoder, test_loader, device)
    nmi_iter, ari_iter, preds, all_results  = evaluate_clustering(Z_test_iter, y_test_iter, base_k=cfg.n_clusters, k_multipliers=(1, 2, 3, 4, 5,6),)
    for k, res in all_results.items():
        print(f"k={k:3d}  NMI={res['nmi']:.3f}  ARI={res['ari']:.3f}")

# ============================
# 4. Final evaluation
# ============================
encoder.eval()
with torch.no_grad():
    Z_test_dc, y_test_dc = extract_embeddings(encoder, test_loader, device)

final_nmi, final_ari, preds, all_results  = evaluate_clustering(Z_test_dc, y_test_dc,base_k=cfg.n_clusters,  k_multipliers=(1, 2, 3, 4, 5,6),)
print(f"\n[Final DeepCluster-lite] NMI = {final_nmi:.4f}, ARI = {final_ari:.4f}")
print(f"[Baseline AE         ] NMI = {base_nmi:.4f}, ARI = {base_ari:.4f}")
print()