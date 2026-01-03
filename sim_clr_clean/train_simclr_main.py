# main.py
import torch
from torch.utils.data import DataLoader
from config import Config
from data.data_access import get_train_test_clips_n_labels, create_label_map
from data.dataset import UrbanSoundDataset
from data.augmentations import get_ae_augment, ContrastiveAug, SafeContrastiveAug, StrongContrastiveAug, \
    make_aug_config, UnifiedAug, TwoViewAug, MixBackground, BackgroundPool
from models.convAE_model import ConvAutoencoder
from models.convVAE_model import ConvVAE
from models.resnetEncoder import ResNetEncoder
from models.ConvAE_SE import ConvAutoencoderSE
from models.simCLRModel import SimCLRModel
from training.trainer_ae import AETrainer, VAETrainer
from training.trainer_simclr import SimCLRrainer
from training.embedding import extract_embeddings, extract_embeddings_VAE, pca_transform
from training.clustering_eval import evaluate_clustering, visualize_embeddings_tsne, per_class_cluster_report, \
    summarize_clustering_quality, kmeans_reject_report, hdbscan_reject_report, purify_train_with_hdbscan
from models.byol_model import BYOLModel
from training.trainer_byol import BYOLTrainer
from models.vicreg_model import VICRegModel
from training.trainer_vicreg import VICRegTrainer
from sklearn.cluster import KMeans

from datetime import datetime
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
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

# aug = SafeContrastiveAug()
# aug = StrongContrastiveAug()
# aug = get_ae_augment() if cfg.model_type in ['ae', 'vae'] else SafeContrastiveAug() #SafeContrastiveAug() #StrongContrastiveAug()
aug = UnifiedAug(make_aug_config(mode=cfg.mode))
cfg_mild = make_aug_config("simclr", target_frames=128)
cfg_strong = make_aug_config("simclr", target_frames=128)
cfg_strong.p_freq_mask = 0.05
cfg_strong.p_time_mask = 0.15
cfg_strong.p_noise = 0.50

# use transient-aware crop inside UnifiedAug
ds = UrbanSoundDataset(    clips=train_clips,    label2id=label2id,
                           precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
                            transform=None,    mode='contrastive',   folds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
bg_pool = BackgroundPool(ds.X, max_items=2048)
mixbg = MixBackground(bg_pool, alpha=(0.03, 0.12), p=0.35)
aug = TwoViewAug(cfg_mild, cfg_strong, shared_crop="transient", mixbg=mixbg)

train_ds = UrbanSoundDataset(
    clips=train_clips,
    label2id=label2id,
    precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
    transform=aug,
    mode='contrastive',  # cfg.model_type,
    folds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    min_confidence=None,
    req_salience=None
)

test_ds = UrbanSoundDataset(
    clips=test_clips,
    label2id=label2id,
    precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
    transform=None,
    mode='ae',
    folds=[10],
    # min_confidence=None,
    # req_salience=None
)

# train_loader = DataLoader(train_ds,
#                          batch_size=cfg.batch_size,
#                          shuffle=True,
#                          pin_memory=True,
#                           )
test_loader = DataLoader(test_ds,
                         batch_size=cfg.batch_size,
                         shuffle=False,
                         pin_memory=True,
                         )

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)

# a deterministic dataset for embedding extraction (no augs)
train_embed_ds = UrbanSoundDataset(
    clips=train_clips,
    label2id=label2id,
    precomputed_file=r"C:\data\soundata_processed\urbansound8k_logmels.pt",
    transform=None,    # <--- no augmentation
    mode="ae",         # <--- returns dict with "x" key; extract_embeddings will pick it
    folds=[0,1,2,3,4,5,6,7,8,9],
)
train_embed_loader = DataLoader(
    train_embed_ds,
    batch_size=cfg.batch_size,
    shuffle=False,     # important: stable ordering
    pin_memory=True,
)


# ======================================================
# Model + Trainer
# ======================================================

# ---- FREEZE early encoder layers ----
def freeze(module):
    for p in module.parameters():
        p.requires_grad = False



# ae = ConvAutoencoder(latent_dim=cfg.latent_dim).to(device)
## state = torch.load(r"C:\data\models\audio_ae_v2_281125.pth", map_location=device)
## ae.load_state_dict(state)
## ae = ConvAutoencoder_v2(latent_dim=cfg.latent_dim, input_size=(1, 64, 126)).to(device)
# state = torch.load(r"C:\data\models\audio_ae_241225.pth", map_location=device)
# ae.load_state_dict(state)

# freeze(ae.enc1)
# freeze(ae.enc2)
resencoder = ResNetEncoder(latent_dim=cfg.latent_dim, depth="layer3", pretrained=cfg.pretrained).to(device)
model = SimCLRModel(ae_backbone=resencoder, latent_dim=cfg.latent_dim).to(device)
# load AE weights into the encoder part
optimizer = torch.optim.Adam([
    {"params": model.encoder.parameters(), "lr": cfg.lr / (5 if cfg.pretrained else 3)},
    {"params": model.projector.parameters(), "lr": cfg.lr},
])
trainer = SimCLRrainer(model, optimizer, device, temperature=0.07, lambda_recon=0.0, amp=True)

warmup_epochs = 5
def warmup_lr(epoch, warmup_epochs=warmup_epochs):
    return min(1.0, (epoch + 1) / warmup_epochs)

warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.epochs-warmup_epochs)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ======================================================
# Train
# ======================================================
purity_all = []
nmi_10 = []
nmi_30 = []
PURIFY_EVERY = 10
MIN_KEEP_FRAC = 0.30   # safety: don’t collapse training set too far

for epoch in range(cfg.epochs):

    # ---- Purify train set (self-supervised) ----
    if epoch > 0 and (epoch % PURIFY_EVERY == 0):
        # keep train_embed_ds aligned with current train_ds subset
        train_embed_ds.set_index_map(getattr(train_ds, "index_map", None))

        kept_real_indices, rep = purify_train_with_hdbscan(
            model=model,
            train_embed_loader=train_embed_loader,
            min_cluster_size=10,
            min_samples=5,
            prob_thresh=0.1,   # knob: raise to reject more
            outlier_q=0.9,     # knob: lower to reject more
            per_cluster_bad=False,
            device=device,
        )

        print(
            f"[PURIFY] kept={rep['kept_frac']*100:.1f}% rejected={rep['rejected_frac']*100:.1f}% "
            f"clusters={rep['n_clusters_excl_noise']} bad_clusters={rep['n_bad_clusters']} "
            f"prob_med={rep['prob_median_all']:.3f}"
        )

        if rep["kept_frac"] >= MIN_KEEP_FRAC:
            train_ds.set_index_map(kept_real_indices)
            # rebuild train_loader so shuffle works on the new subset
            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                pin_memory=True,
            )
        else:
            print("[PURIFY] skip update (kept_frac too small)")


    train_loss = trainer.train_epoch(train_loader, epoch)

    # if epoch < warmup_epochs:
    #     warmup.step()
    # else:
    scheduler.step()

    print(f"[Epoch {epoch+1}] LR = {scheduler.get_last_lr()[0]:.6f}, Loss = {train_loss:.4f}")

    if epoch % 10 ==0:
        Z, labels = extract_embeddings(model.encoder, test_loader, device, normalize=False)

        Z_pca, exp_var = pca_transform(Z, n_components=36, whiten=False, l2_after=True)
        print(f"explained variance: {sum(exp_var)}")

        _, _, preds, all_results = evaluate_clustering(
            Z_pca,
            labels,
            base_k=len(label2id),
            method="kmeans",
            k_multipliers=(1, 1.5, 2, 2.5, 3),
        )

            #~~~~~~~ cluster eval  ~~~~~~
        X = Z_pca  # or normalized encoder embeddings
        # if you want cosine silhouette, normalize first:
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        sil = silhouette_score(Xn, preds, metric="cosine")
        db = davies_bouldin_score(Xn, preds)  # uses euclidean internally; ok on normalized
        ch = calinski_harabasz_score(Xn, preds)
        print(f'clustering score: {sil}, {ch}, {db}')
            #~~~~~~~~~~~~~~~~
        # choose best k under a cap (e.g. ≤30 clusters)
        MAX_K = 30
        best = None
        best_score = -1

        for k, res in all_results.items():
            if k > MAX_K:
                continue
            summary = summarize_clustering_quality(labels, res["preds"])
            score = summary["weighted_purity"] * summary["frac≥0.70"]
            if score > best_score:
                best_score = score
                best = (k, summary)

        k, s = best
        print(
            f"[k={k:2d}] "
            f"purity_w={s['weighted_purity']:.3f} | "
            f"frac≥0.70={s['frac≥0.70']:.3f} | "
            f"frac≥0.80={s['frac≥0.80']:.3f} | "
            f"eff_cls={s['effective_clusters']:.2f}"
        )
        for k, res in all_results.items():
            print(f"k={k:3d}  NMI={res['nmi']:.3f}  ARI={res['ari']:.3f}")
        purity_all.append(s['weighted_purity'])
        nmi_10.append(all_results[10])
        nmi_30.append(all_results[30])

        k_report = 20
        preds = all_results[k_report]["preds"]

        # re-fit to get centers (only needed if you don't already have them)
        km = KMeans(n_clusters=k_report, n_init=20, random_state=0).fit(Xn)
        # If you want to force using the same preds from all_results, keep preds=...,
        # but still use km.cluster_centers_ for distances.

        preds_other, keep_mask, cluster_df, summ = kmeans_reject_report(
            X=Xn,
            labels_true=labels,
            preds_kmeans=preds,
            kmeans_centers=km.cluster_centers_,
            reject_frac=0.2,  # << knob #1 (try 0.15..0.40)
            margin_thresh=0.05,  # << knob #2 (try 0.05..0.20)
            use_margin=True,
            per_cluster_bad=True,  # << optional cluster-level kill switch
            bad_z_thresh=3.0,
            bad_kept_frac_thresh=0.35,
            min_cluster_size=10,
        )

        print(
            f"[k={k_report}] kept={summ['kept_frac'] * 100:.1f}%  rejected={summ['rejected_frac'] * 100:.1f}%  "
            f"NMI_all={summ['nmi_all']:.3f}  NMI_kept={summ['nmi_kept']:.3f}  "
            f"bad_clusters={summ['n_bad_clusters']}"
        )

        # show top “worst” clusters (most rejected / bad)
        # print(cluster_df.head(12).to_string(index=False))

        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        preds_hdb_other, keep_mask, cluster_df, summ, hdb = hdbscan_reject_report(
            X=Xn,
            labels_true=labels,
            min_cluster_size=5,  # start around ~0.5–1.5x your "expected event cluster size"
            min_samples=5,  # raise => more conservative (more rejected)
            metric="euclidean",
            cluster_selection_method="eom",
            prob_thresh=0.03,  # sensitivity: 0.0 disables
            outlier_q=0.999,  # sensitivity: reject top 2% outliers among non-noise
            per_cluster_bad=False,
            bad_kept_frac_thresh=0.35,
            bad_median_prob_thresh=0.20,
            cluster_selection_epsilon=0.05
        )
        print(
            f"[HDBSCAN] clusters={summ['n_clusters_excluding_noise']}  "
            f"kept={summ['kept_frac'] * 100:.1f}%  rejected={summ['rejected_frac'] * 100:.1f}%  "
            f"NMI_all={summ['nmi_all']:.3f}  NMI_kept={summ['nmi_kept']:.3f}  "
            f"bad_clusters={summ['n_bad_clusters']}"
        )
        # Per-cluster info (top 15 “best kept” non-noise clusters)
        # print(cluster_df[cluster_df["is_noise"] == False].head(15).to_string(index=False))
        # Optionally: NMI on original “all points but noise only”, if you want a different definition:
        noise_only = (hdb.labels_ == -1)
        print("noise_frac:", noise_only.mean())

date_str = str(datetime.now())[:-7].replace(':', '_')
torch.save(model.encoder.state_dict(), rf"C:\data\models\simclr_res{cfg.latent_dim}_{date_str}.pth")
visualize_embeddings_tsne(model.encoder, test_loader, r"C:\tmp\encoded_tsne2.html", perplexity=12)
