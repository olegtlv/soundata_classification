# training/clustering_eval.py
from sklearn.cluster import KMeans, SpectralClustering
import torch
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import hdbscan
from training.embedding import extract_embeddings, pca_transform


def clustering_reports(
    labels,
    preds,
    label_names=None,          # optional list of class names by id
    top_labels=5,              # how many labels to show in summaries
    normalize_rows=True,       # normalize confusion by true label (row-wise)
):
    """
    labels: [N] ground truth int labels
    preds : [N] cluster ids (0..k-1)
    Returns:
      cm_df: confusion matrix (true label x cluster)
      cm_row_norm_df: row-normalized confusion (true label distribution over clusters)
      cluster_comp_df: per-cluster composition (% per label + purity)
      cluster_sizes_df: cluster sizes
    """
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)

    uniq_labels = np.unique(labels)
    uniq_clusters = np.unique(preds)

    # nice names
    if label_names is None:
        label_names = [str(i) for i in uniq_labels]
        label_id_to_name = {i: str(i) for i in uniq_labels}
    else:
        label_id_to_name = {i: label_names[i] for i in uniq_labels}

    # --- confusion counts: rows=true labels, cols=clusters
    cm = pd.crosstab(
        pd.Series(labels).map(label_id_to_name),
        pd.Series(preds).map(lambda c: f"cluster_{c}"),
        dropna=False
    )

    # ensure all clusters appear
    for c in uniq_clusters:
        col = f"cluster_{c}"
        if col not in cm.columns:
            cm[col] = 0
    cm = cm.reindex(sorted(cm.columns, key=lambda x: int(x.split("_")[1])), axis=1)

    # row-normalized version (distribution of each true class across clusters)
    if normalize_rows:
        cm_row = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    else:
        cm_row = None

    # --- per-cluster label composition (% within cluster)
    cluster_rows = []
    for c in uniq_clusters:
        mask = preds == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue

        labs_c = labels[mask]
        counts = pd.Series(labs_c).value_counts().sort_values(ascending=False)
        # map ids to names
        counts_named = counts.rename(index=label_id_to_name)

        purity = float(counts.max() / n_c)
        top = (counts_named / n_c * 100.0).head(top_labels)

        row = {
            "cluster": int(c),
            "size": n_c,
            "purity": purity,
            "majority_label": str(counts_named.index[0]),
            "majority_pct": float(counts.max() / n_c * 100.0),
        }
        # add top label % columns
        for i, (lab, pct) in enumerate(top.items(), start=1):
            row[f"top{i}_label"] = lab
            row[f"top{i}_pct"] = float(pct)

        # also: full distribution as a dict (handy for printing/debug)
        full_dist = (counts_named / n_c * 100.0).to_dict()
        row["label_pct_dict"] = full_dist

        cluster_rows.append(row)

    cluster_comp_df = pd.DataFrame(cluster_rows).sort_values(
        by=["purity", "size"], ascending=[False, False]
    ).reset_index(drop=True)

    cluster_sizes_df = pd.DataFrame({
        "cluster": [int(c) for c in uniq_clusters],
        "size": [int((preds == c).sum()) for c in uniq_clusters]
    }).sort_values("size", ascending=False).reset_index(drop=True)

    return cm, cm_row, cluster_comp_df, cluster_sizes_df


def evaluate_clustering_with_reports(
    Z,
    labels,
    base_k,
    method="kmeans",
    k_multipliers=(1,),
    random_state=42,
    label_names=None,
    top_labels=5,
):
    """
    Calls your evaluate_clustering(...) and adds reports per k.
    Returns:
      nmi_base, ari_base, preds_base, all_results, reports_by_k
    where reports_by_k[k] = {
        "confusion": cm_df,
        "confusion_row_norm": cm_row_df,
        "cluster_composition": cluster_comp_df,
        "cluster_sizes": cluster_sizes_df,
    }
    """
    nmi_base, ari_base, preds_base, all_results = evaluate_clustering(
        Z=Z,
        labels=labels,
        base_k=base_k,
        method=method,
        k_multipliers=k_multipliers,
        random_state=random_state,
    )

    reports_by_k = {}
    for k, res in all_results.items():
        preds = res["preds"]
        cm_df, cm_row_df, cluster_comp_df, cluster_sizes_df = clustering_reports(
            labels=labels,
            preds=preds,
            label_names=label_names,
            top_labels=top_labels,
            normalize_rows=True,
        )
        reports_by_k[int(k)] = {
            "confusion": cm_df,
            "confusion_row_norm": cm_row_df,
            "cluster_composition": cluster_comp_df,
            "cluster_sizes": cluster_sizes_df,
        }

    return nmi_base, ari_base, preds_base, all_results, reports_by_k


def evaluate_clustering(
    Z,
    labels,
    base_k,
    method: str = "kmeans",
    k_multipliers=(1,),  # e.g. (1, 2, 3) -> k, 2k, 3k
    random_state: int = 42,
):
    """
    Evaluate clustering quality (NMI, ARI) for one or more k values.

    Parameters
    ----------
    Z : array-like, shape [n_samples, n_features]
        Embeddings.
    labels : array-like, shape [n_samples]
        Ground-truth labels.
    base_k : int
        "Reference" number of clusters (usually #classes = len(label2id)).
    method : {"kmeans", "gmm", "spectral"}
        Clustering method to use.
    k_multipliers : iterable of int or float
        Factors to multiply `base_k` by. For example:
          - (1,)     -> just base_k (backwards compatible)
          - (1, 2, 3)-> base_k, 2*base_k, 3*base_k
        If a factor is a float, k = int(round(base_k * factor)).
        If it's an int > base_k, treated as absolute k.
    random_state : int
        Random seed where applicable.

    Returns
    -------
    nmi_base : float
        NMI for k = base_k (for backwards-compat).
    ari_base : float
        ARI for k = base_k (for backwards-compat).
    preds_base : np.ndarray
        Cluster assignments for k = base_k (for backwards-compat).
    all_results : dict
        Dictionary keyed by k:
        {
          k: {
            "nmi": float,
            "ari": float,
            "preds": np.ndarray of shape [n_samples]
          },
          ...
        }
    """
    Z = np.asarray(Z)
    labels = np.asarray(labels)

    all_results = {}

    # normalize multipliers into concrete k values
    ks = set()
    for m in k_multipliers:
        if isinstance(m, float):
            k = int(round(base_k * m))
        else:
            # if it's an int < base_k, treat as factor; if >= base_k, treat as absolute k
            if m <= 0:
                continue
            if m < base_k:
                k = int(round(base_k * m))
            else:
                k = int(m)
        if k >= 2:  # at least 2 clusters
            ks.add(k)

    # always ensure base_k is included
    ks.add(int(base_k))
    ks = sorted(ks)

    for k in ks:
        if method == "kmeans":
            clusterer = KMeans(n_clusters=k, random_state=random_state)
            preds = clusterer.fit_predict(Z)

        elif method == "gmm":
            clusterer = GaussianMixture(
                n_components=k,
                covariance_type="tied",
                random_state=random_state,
            )
            preds = clusterer.fit(Z).predict(Z)

        elif method == "spectral":
            clusterer = SpectralClustering(
                n_clusters=k,
                affinity="nearest_neighbors",
                assign_labels="kmeans",
                random_state=random_state,
            )
            preds = clusterer.fit_predict(Z)

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        nmi = normalized_mutual_info_score(labels, preds)
        ari = adjusted_rand_score(labels, preds)

        all_results[k] = {
            "nmi": float(nmi),
            "ari": float(ari),
            "preds": preds,
        }

    # Backwards-compatible return for base_k
    base_k = int(base_k)
    base_res = all_results[base_k]
    return base_res["nmi"], base_res["ari"], base_res["preds"], all_results


def visualize_embeddings_tsne(model, loader, output_html_path, perplexity=5):
    """
    Encodes data from loader using model, performs t-SNE, KMeans clustering,
    computes NMI/ARI, and saves an interactive Plotly HTML.

    Args:
        model: torch.nn.Module autoencoder or embedding model.
        loader: DataLoader yielding dicts with keys 'x_clean' and 'label'.
        output_html_path: str, path to save Plotly HTML.
        device: 'cpu' or 'cuda'.
        perplexity: t-SNE perplexity.
    """
    model.eval()
    encoded, categories = [], []

    with torch.no_grad():
        for batch in loader:
            device = next(model.parameters()).device  # get device of model
            x = batch['x_clean'].to(device).float()
            _, z = model(x)
            encoded.append(z.cpu())
            categories.append(batch['label'].cpu())

    encoded = torch.cat(encoded, dim=0)
    categories = torch.cat(categories, dim=0)

    X = encoded.numpy()
    y = categories.numpy()

    # --- t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_2d = tsne.fit_transform(X)

    # --- DataFrame for Plotly
    df = pd.DataFrame({
        'x': X_2d[:, 0],
        'y': X_2d[:, 1],
        'category': y
    })

    # --- Plot
    fig = px.scatter(
        df,
        x='x', y='y',
        color=df['category'].astype(str),
        title="t-SNE visualization of embeddings (colored by category)",
        opacity=0.7,
        template="plotly_dark"
    )

    fig.write_html(output_html_path)
    print(f"✅ Saved interactive visualization to: {output_html_path}")
    return fig

def per_class_cluster_report(y_true, y_pred, id2label=None):
    """
    y_true: 1D array-like of true class IDs
    y_pred: 1D array-like of cluster IDs (k-means assignments)
    id2label: optional dict {class_id: class_name}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    clusters = np.unique(y_pred)

    print("Per-class clustering report:")
    print("class\tcluster\tcount\tprec\trecall\tF1")

    for c in classes:
        mask_c = (y_true == c)
        n_c = mask_c.sum()
        if n_c == 0:
            continue

        # distribution of this class over clusters
        c_clusters, c_counts = np.unique(y_pred[mask_c], return_counts=True)

        # dominant cluster for this class
        best_cluster = c_clusters[np.argmax(c_counts)]
        tp = c_counts.max()  # # of class-c samples in best_cluster

        # cluster size
        n_cluster = (y_pred == best_cluster).sum()

        recall = tp / n_c
        precision = tp / n_cluster if n_cluster > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        name = id2label[c] if id2label is not None else str(c)
        print(f"{name}\t{best_cluster}\t{tp}/{n_c}\t{precision:.3f}\t{recall:.3f}\t{f1:.3f}")


def cluster_purity_report(
    y_true,
    y_cluster,
    label_names=None,          # optional: list/dict to map class ids -> names
    min_cluster_size=1,        # ignore tiny clusters if you want (e.g. 5)
    purity_thresholds=(0.7, 0.8, 0.9),
):
    """
    Computes:
      1) Per-cluster purity + dominant label
      2) Weighted purity (overall)
      3) Per-class coverage: how many clusters each class occupies and how pure those clusters are

    y_true:   [N] ground-truth class ids (ints)
    y_cluster:[N] cluster assignment ids (ints)
    """

    y_true = np.asarray(y_true)
    y_cluster = np.asarray(y_cluster)

    # Confusion (rows=true labels, cols=clusters)
    df_conf = pd.crosstab(y_true, y_cluster)

    # Optionally rename rows
    if label_names is not None:
        if isinstance(label_names, dict):
            df_conf.index = [label_names.get(i, i) for i in df_conf.index]
        else:
            # list-like
            df_conf.index = [label_names[i] if i < len(label_names) else i for i in df_conf.index]

    # Cluster sizes (by column)
    cluster_sizes = df_conf.sum(axis=0)

    # Per-cluster purity
    dom_counts = df_conf.max(axis=0)
    dom_label = df_conf.idxmax(axis=0)
    purity = (dom_counts / cluster_sizes.replace(0, np.nan)).fillna(0.0)

    per_cluster = pd.DataFrame({
        "size": cluster_sizes,
        "dominant_label": dom_label,
        "dominant_count": dom_counts,
        "purity": purity,
    }).sort_values(["purity", "size"], ascending=[False, False])

    # Apply min_cluster_size filter for summaries (keep df_conf full)
    mask_keep = per_cluster["size"] >= int(min_cluster_size)
    per_cluster_kept = per_cluster[mask_keep]

    # Weighted purity: sum over clusters of (cluster_size * purity) / total_size
    total = per_cluster_kept["size"].sum()
    weighted_purity = float((per_cluster_kept["size"] * per_cluster_kept["purity"]).sum() / max(total, 1))

    # Purity bucket counts (by number of samples, weighted)
    buckets = {}
    for thr in purity_thresholds:
        s = per_cluster_kept.loc[per_cluster_kept["purity"] >= thr, "size"].sum()
        buckets[f"frac_samples_in_purity>={thr:.2f}"] = float(s / max(total, 1))

    # Per-class coverage:
    # For each true class row, look at distribution across clusters.
    # Define "clusters used" as clusters where class has >0 samples.
    # Also compute how many of those clusters are "pure-ish" for that class (dominant label = class and purity>=thr).
    per_class_rows = []
    # We need the class ids / names exactly as in df_conf.index
    class_index = df_conf.index.tolist()

    for cls in class_index:
        row = df_conf.loc[cls]  # counts across clusters
        used_cols = row[row > 0].index.tolist()
        num_used = len(used_cols)
        cls_total = int(row.sum())

        # For those clusters, compute the cluster purity and whether class is dominant there
        cls_cluster_info = per_cluster.loc[used_cols].copy()
        cls_cluster_info["cls_count_in_cluster"] = row[used_cols].values
        cls_cluster_info["cls_frac_of_cluster"] = cls_cluster_info["cls_count_in_cluster"] / cls_cluster_info["size"].replace(0, np.nan)
        cls_cluster_info["cls_frac_of_class"] = cls_cluster_info["cls_count_in_cluster"] / max(cls_total, 1)

        # summary stats
        # effective number of clusters (entropy-based) = exp(H(p)), where p is fraction of class across clusters
        p = np.asarray(cls_cluster_info["cls_frac_of_class"].values, dtype=np.float64)
        p = p[p > 0]
        H = -np.sum(p * np.log(p + 1e-12))
        eff_clusters = float(np.exp(H)) if p.size else 0.0

        # how many clusters where this class is the dominant label
        dom_count = int((cls_cluster_info["dominant_label"] == cls).sum())

        # “pure-ish” clusters for this class under thresholds
        pure_counts = {}
        for thr in purity_thresholds:
            pure_counts[f"n_clusters_dom_and_purity>={thr:.2f}"] = int(
                ((cls_cluster_info["dominant_label"] == cls) & (cls_cluster_info["purity"] >= thr)).sum()
            )

        per_class_rows.append({
            "class": cls,
            "n_samples": cls_total,
            "n_clusters_used": num_used,
            "effective_clusters": eff_clusters,
            "n_clusters_where_class_is_dominant": dom_count,
            **pure_counts,
            # concentration: share of class in its top-1 / top-2 clusters
            "top1_cluster_share_of_class": float(np.sort(p)[-1]) if p.size else 0.0,
            "top2_cluster_share_of_class": float(np.sort(p)[-2:].sum()) if p.size >= 2 else (float(np.sort(p)[-1]) if p.size else 0.0),
        })

    per_class = pd.DataFrame(per_class_rows).sort_values("n_samples", ascending=False)

    return {
        "confusion": df_conf,
        "per_cluster": per_cluster,
        "per_cluster_kept": per_cluster_kept,
        "weighted_purity": weighted_purity,
        "purity_buckets": buckets,
        "per_class": per_class,
    }


# -------------------------
# Example usage
# -------------------------
# Suppose you have:
#   labels = ground truth labels for the set you clustered (same order as Z)
#   preds  = cluster assignments from evaluate_clustering(...)

# report = cluster_purity_report(labels, preds, label_names=None, min_cluster_size=5, purity_thresholds=(0.7,0.8,0.9))
# print("Weighted purity:", report["weighted_purity"])
# print("Purity buckets:", report["purity_buckets"])
# display(report["per_cluster"].head(10))
# display(report["per_class"])
# display(report["confusion"])



import numpy as np
import pandas as pd

def summarize_clustering_quality(
    labels,
    preds,
    max_clusters=None,
    purity_thresholds=(0.7, 0.8),
):
    """
    Returns a compact dict with purity-focused metrics.
    """
    df = pd.DataFrame({"y": labels, "c": preds})

    # cluster sizes
    cluster_sizes = df["c"].value_counts()
    n_clusters = len(cluster_sizes)

    # cluster purity
    cluster_stats = []
    for c, g in df.groupby("c"):
        counts = g["y"].value_counts()
        dominant = counts.iloc[0]
        purity = dominant / len(g)
        cluster_stats.append({
            "cluster": c,
            "size": len(g),
            "purity": purity,
        })

    stats = pd.DataFrame(cluster_stats)

    # weighted purity
    weighted_purity = np.average(stats["purity"], weights=stats["size"])

    # purity coverage
    frac_by_thr = {
        f"frac≥{thr:.2f}": stats.loc[stats["purity"] >= thr, "size"].sum() / len(df)
        for thr in purity_thresholds
    }

    # effective clusters per class
    eff_clusters = []
    for y, g in df.groupby("y"):
        p = g["c"].value_counts(normalize=True)
        eff = np.exp(-(p * np.log(p + 1e-9)).sum())
        eff_clusters.append(eff)
    eff_clusters = float(np.mean(eff_clusters))

    return {
        "n_clusters": n_clusters,
        "weighted_purity": float(weighted_purity),
        "effective_clusters": eff_clusters,
        **frac_by_thr,
    }


import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

def _l2n(X, eps=1e-12):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def _robust_mad(x, eps=1e-12):
    med = np.median(x)
    return np.median(np.abs(x - med)) + eps

def kmeans_reject_report(
    X,                      # [N,D] (use Xn = l2-normalized PCA output)
    labels_true,            # [N] true labels (only for NMI reporting)
    preds_kmeans,           # [N] kmeans hard assignments for some k
    kmeans_centers,         # [K,D] centers from the same kmeans fit
    reject_frac=0.30,       # knob #1: global reject by distance tail (0..1)
    margin_thresh=0.10,     # knob #2: reject if too ambiguous between 1st/2nd center
    use_margin=True,
    per_cluster_bad=True,   # optional: mark whole clusters as bad if very “noisy”
    min_cluster_size=12,
    bad_z_thresh=3.0,       # cluster is bad if its median dist is > global_med + z*MAD
    bad_kept_frac_thresh=0.35,  # cluster is bad if it keeps too few points
    other_label=-1,
):
    """
    Returns:
      preds_with_other: [N] where rejected samples are labeled as -1
      keep_mask: [N] bool
      cluster_df: per-cluster stats DataFrame
      summary: dict with rates + NMI(all/kept)
    """

    X = np.asarray(X)
    y = np.asarray(labels_true)
    preds = np.asarray(preds_kmeans).astype(int)
    C = np.asarray(kmeans_centers)

    N = X.shape[0]
    K = C.shape[0]

    # distances to all centers: [N,K]
    d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)

    # assigned distance
    d1 = d[np.arange(N), preds]

    # margin (how clearly it prefers its best center over 2nd best)
    d_sorted = np.sort(d, axis=1)
    best = d_sorted[:, 0]
    second = d_sorted[:, 1]
    margin = (second - best) / (second + 1e-12)  # 0 = very ambiguous, larger = clearer

    # --- sample-level rejection ---
    keep = np.ones(N, dtype=bool)

    # (A) distance-tail reject
    if reject_frac is not None and reject_frac > 0:
        thr = np.quantile(d1, 1.0 - reject_frac)
        keep &= (d1 <= thr)
    else:
        thr = None

    # (B) margin reject
    if use_margin and margin_thresh is not None:
        keep &= (margin >= margin_thresh)

    # --- cluster-level “bad cluster” detection (optional) ---
    bad_clusters = set()
    if per_cluster_bad:
        g_med = np.median(d1)
        g_mad = _robust_mad(d1)

        # compute kept fraction per cluster first
        kept_frac_by_c = np.zeros(K, dtype=float)
        for c in range(K):
            idx = np.where(preds == c)[0]
            if len(idx) == 0:
                kept_frac_by_c[c] = 0.0
            else:
                kept_frac_by_c[c] = keep[idx].mean()

        for c in range(K):
            idx = np.where(preds == c)[0]
            if len(idx) < min_cluster_size:
                bad_clusters.add(c)
                continue
            dc = d1[idx]
            z = (np.median(dc) - g_med) / g_mad
            if z > bad_z_thresh:
                bad_clusters.add(c)
            if kept_frac_by_c[c] < bad_kept_frac_thresh:
                bad_clusters.add(c)

        # if a cluster is bad => reject all its points
        if len(bad_clusters) > 0:
            keep &= ~np.isin(preds, list(bad_clusters))

    # build preds_with_other
    preds_with_other = preds.copy()
    preds_with_other[~keep] = other_label

    # --- NMI reporting ---
    nmi_all = normalized_mutual_info_score(y, preds)  # plain kmeans
    if keep.sum() > 0:
        nmi_kept = normalized_mutual_info_score(y[keep], preds[keep])
    else:
        nmi_kept = np.nan

    # --- per-cluster stats ---
    rows = []
    for c in range(K):
        idx = np.where(preds == c)[0]
        if len(idx) == 0:
            continue
        dc = d1[idx]
        mc = margin[idx]
        kept_c = keep[idx]

        rows.append({
            "cluster": c,
            "size": int(len(idx)),
            "kept_frac": float(kept_c.mean()),
            "kept_n": int(kept_c.sum()),
            "median_dist": float(np.median(dc)),
            "mad_dist": float(_robust_mad(dc)),
            "median_margin": float(np.median(mc)),
            "bad_cluster": bool(c in bad_clusters),
        })

    cluster_df = pd.DataFrame(rows).sort_values(
        ["bad_cluster", "kept_frac", "size"],
        ascending=[True, False, False]
    )

    summary = {
        "N": int(N),
        "K": int(K),
        "rejected_frac": float((~keep).mean()),
        "kept_frac": float(keep.mean()),
        "distance_thr": None if thr is None else float(thr),
        "margin_thresh": None if margin_thresh is None else float(margin_thresh),
        "nmi_all": float(nmi_all),
        "nmi_kept": float(nmi_kept) if np.isfinite(nmi_kept) else np.nan,
        "n_bad_clusters": int(len(bad_clusters)),
    }

    return preds_with_other, keep, cluster_df, summary


import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

def _l2n(X, eps=1e-12):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def hdbscan_reject_report(
    X,                      # [N,D] embeddings (prefer normalized, see usage)
    labels_true,            # [N] true labels (only used for NMI reporting)
    min_cluster_size=30,    # knob #1 (bigger => fewer, cleaner clusters + more rejected)
    min_samples=None,       # knob #2 (bigger => more conservative => more rejected)
    metric="euclidean",     # with X normalized, euclidean ~= cosine-ish
    cluster_selection_method="eom",  # "eom" or "leaf" ("leaf" => more clusters)
    prob_thresh=0.0,        # knob #3: reject members with low membership probability (0..1)
    outlier_q=None,         # knob #4: reject top-q outliers within clustered points (e.g. 0.95)
    per_cluster_bad=True,
    min_cluster_size_for_stats=12,
    bad_kept_frac_thresh=0.35,
    bad_median_prob_thresh=0.20,
    cluster_selection_epsilon=0.05,  # try 0.02, 0.05, 0.1
    other_label=-1,

):
    """
    Returns:
      preds_with_other: [N] (rejected labeled as -1)
      keep_mask: [N] bool
      cluster_df: per-cluster stats DataFrame
      summary: dict with rejected% + NMI(all/kept)
      hdb: fitted HDBSCAN object (for debugging/plots)
    """

    # --- import here so your file still runs without it installed ---
    import hdbscan

    X = np.asarray(X)
    y = np.asarray(labels_true)

    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        prediction_data=False,
        cluster_selection_epsilon=cluster_selection_epsilon  # try 0.02, 0.05, 0.1

    ).fit(X)

    preds = hdb.labels_.copy()               # -1 means noise
    prob = getattr(hdb, "probabilities_", None)   # [N] in [0,1]
    outl = getattr(hdb, "outlier_scores_", None)  # [N], larger = more outlier-ish (only meaningful for non-noise too)

    N = len(preds)
    keep = (preds != other_label)

    # (A) membership-prob rejection (optional)
    if prob is not None and prob_thresh is not None and prob_thresh > 0:
        keep &= (prob >= prob_thresh)

    # (B) outlier-score rejection (optional)
    # We compute a threshold over the currently-kept points so the quantile is meaningful.
    if outl is not None and outlier_q is not None:
        kept_idx = np.where(keep)[0]
        if len(kept_idx) > 0:
            thr = np.quantile(outl[kept_idx], outlier_q)
            keep &= (outl <= thr)
        else:
            thr = None
    else:
        thr = None

    # (C) cluster-level “bad cluster” marking (optional)
    bad_clusters = set()
    if per_cluster_bad:
        cluster_ids = [c for c in np.unique(preds) if c != other_label]
        for c in cluster_ids:
            idx = np.where(preds == c)[0]
            if len(idx) < min_cluster_size_for_stats:
                bad_clusters.add(int(c))
                continue

            kept_frac_c = keep[idx].mean()
            med_prob_c = float(np.median(prob[idx])) if prob is not None else 1.0

            if kept_frac_c < bad_kept_frac_thresh:
                bad_clusters.add(int(c))
            if med_prob_c < bad_median_prob_thresh:
                bad_clusters.add(int(c))

        if len(bad_clusters) > 0:
            keep &= ~np.isin(preds, list(bad_clusters))

    preds_with_other = preds.copy()
    preds_with_other[~keep] = other_label

    # --- NMI reporting ---
    # Note: NMI_all includes noise as a "cluster". That's okay, but interpret it accordingly.
    nmi_all = normalized_mutual_info_score(y, preds_with_other)

    if keep.sum() > 0:
        nmi_kept = normalized_mutual_info_score(y[keep], preds[keep])
    else:
        nmi_kept = np.nan

    # --- per-cluster stats table ---
    rows = []
    cluster_ids = np.unique(preds)
    for c in cluster_ids:
        idx = np.where(preds == c)[0]
        if len(idx) == 0:
            continue

        rows.append({
            "cluster": int(c),
            "is_noise": bool(c == other_label),
            "size": int(len(idx)),
            "kept_frac": float(keep[idx].mean()),
            "kept_n": int(keep[idx].sum()),
            "median_prob": float(np.median(prob[idx])) if prob is not None else np.nan,
            "median_outlier": float(np.median(outl[idx])) if outl is not None else np.nan,
            "bad_cluster": bool(int(c) in bad_clusters) if c != other_label else False,
        })

    cluster_df = pd.DataFrame(rows).sort_values(
        ["is_noise", "bad_cluster", "kept_frac", "size"],
        ascending=[True, True, False, False]
    )

    summary = {
        "N": int(N),
        "n_clusters_excluding_noise": int(len([c for c in np.unique(preds) if c != other_label])),
        "rejected_frac": float((~keep).mean()),
        "kept_frac": float(keep.mean()),
        "min_cluster_size": int(min_cluster_size),
        "min_samples": None if min_samples is None else int(min_samples),
        "prob_thresh": float(prob_thresh),
        "outlier_q": None if outlier_q is None else float(outlier_q),
        "outlier_thr": None if thr is None else float(thr),
        "n_bad_clusters": int(len(bad_clusters)),
        "nmi_all": float(nmi_all),
        "nmi_kept": float(nmi_kept) if np.isfinite(nmi_kept) else np.nan,
    }

    return preds_with_other, keep, cluster_df, summary, hdb

def purify_train_with_hdbscan(
    model,
    train_embed_loader,
    min_cluster_size=15,
    min_samples=15,
    metric="euclidean",
    prob_thresh=0.05,       # knob 1: higher => more rejected
    outlier_q=0.98,         # knob 2: lower => more rejected
    per_cluster_bad=True,   # knob 3: drop entire “bad clusters”
    bad_median_prob_thresh=0.15,
    bad_kept_frac_thresh=0.35,
    device="cuda",
):
    """
    Returns:
      kept_real_indices (np.ndarray of real indices into train_ds.X)
      report dict
    """
    model.eval()

    # 1) embeddings from current encoder
    Z, _ = extract_embeddings(model.encoder, train_embed_loader, device, normalize=False)

    # 2) PCA + L2 (your current evaluation style)
    Z_pca, _ = pca_transform(Z, n_components=48, whiten=False, l2_after=True)

    # 3) normalize for “spherical” geometry (recommended for euclidean HDBSCAN here)
    Xn = Z_pca / (np.linalg.norm(Z_pca, axis=1, keepdims=True) + 1e-12)

    # 4) cluster
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels_h = clusterer.fit_predict(Xn)               # -1 = noise
    probs = clusterer.probabilities_.astype(np.float32)  # [N]

    # 5) point-level rejection
    keep = (labels_h != -1)
    if prob_thresh is not None and prob_thresh > 0:
        keep &= (probs >= prob_thresh)

    # outlier-score rejection among non-noise
    if outlier_q is not None and 0 < outlier_q < 1:
        out_s = clusterer.outlier_scores_.astype(np.float32)
        thr = np.quantile(out_s[labels_h != -1], outlier_q) if np.any(labels_h != -1) else np.inf
        keep &= (out_s <= thr)

    # 6) optional: drop whole bad clusters
    bad_clusters = set()
    if per_cluster_bad:
        for c in np.unique(labels_h):
            if c == -1:
                continue
            m = (labels_h == c)
            frac_kept = float(keep[m].mean()) if m.any() else 0.0
            med_prob = float(np.median(probs[m])) if m.any() else 0.0
            if (med_prob < bad_median_prob_thresh) or (frac_kept < bad_kept_frac_thresh):
                bad_clusters.add(int(c))
        if bad_clusters:
            keep &= ~np.isin(labels_h, list(bad_clusters))

    report = {
        "n_total": int(len(labels_h)),
        "kept_frac": float(keep.mean()),
        "rejected_frac": float((~keep).mean()),
        "n_clusters_excl_noise": int(len(set(labels_h.tolist())) - (1 if -1 in labels_h else 0)),
        "n_bad_clusters": int(len(bad_clusters)),
        "prob_median_all": float(np.median(probs)),
    }

    # 7) IMPORTANT: map back to real indices from dataset
    # We rely on train_embed_loader.dataset providing 'index' field in batches
    # If your extract_embeddings doesn’t return indices, we do it directly from the dataset ordering:
    # -> kept indices are in [0..N-1] of train_embed_loader.dataset, then map via dataset.index_map if exists.
    ds = train_embed_loader.dataset
    if getattr(ds, "index_map", None) is not None:
        real_indices = ds.index_map.cpu().numpy()
    else:
        real_indices = np.arange(len(ds), dtype=np.int64)

    kept_real_indices = real_indices[keep]
    return kept_real_indices, report



# assumes you already have these:
# - extract_embeddings
# - pca_transform
# - kmeans_reject_report  (your earlier helper that returns keep_mask etc.)

def purify_train_with_kmeans(
    model,
    train_embed_loader,
    k=20,                   # number of clusters to fit for purification (try 10..50)
    n_init=20,
    seed=0,
    # knobs:
    reject_frac=0.25,       # reject farthest X% within each cluster (0.15..0.40)
    margin_thresh=0.05,     # reject ambiguous points (0.03..0.20)
    use_margin=True,
    per_cluster_bad=True,   # optionally drop whole clusters deemed "bad"
    bad_z_thresh=3.0,
    bad_kept_frac_thresh=0.35,
    min_cluster_size=10,    # ignore cluster-level decisions for tiny clusters
    device="cuda",
    pca_components=48,
):
    """
    Returns:
      kept_real_indices: np.ndarray of indices into the *real* precomputed tensor ds.X
      report: dict with kept/rejected fractions and some diagnostics
    """
    model.eval()

    # 1) embeddings from current encoder
    Z, labels_true = extract_embeddings(model.encoder, train_embed_loader, device, normalize=False)

    # 2) PCA + L2 (same style you use elsewhere)
    Z_pca, _ = pca_transform(Z, n_components=pca_components, whiten=False, l2_after=True)

    # 3) L2 normalize to "sphere" so euclidean ≈ cosine geometry
    Xn = Z_pca / (np.linalg.norm(Z_pca, axis=1, keepdims=True) + 1e-12)

    # 4) kmeans
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    preds = km.fit_predict(Xn)
    centers = km.cluster_centers_

    # 5) rejection (point-level + optional cluster-level badness)
    preds_other, keep_mask, cluster_df, summ = kmeans_reject_report(
        X=Xn,
        labels_true=labels_true,          # not used for the rejection itself; only for reporting if you pass it
        preds_kmeans=preds,
        kmeans_centers=centers,
        reject_frac=reject_frac,
        margin_thresh=margin_thresh,
        use_margin=use_margin,
        per_cluster_bad=per_cluster_bad,
        bad_z_thresh=bad_z_thresh,
        bad_kept_frac_thresh=bad_kept_frac_thresh,
        min_cluster_size=min_cluster_size,
    )

    report = {
        "n_total": int(len(keep_mask)),
        "kept_frac": float(np.mean(keep_mask)),
        "rejected_frac": float(1.0 - np.mean(keep_mask)),
        "k": int(k),
        "n_bad_clusters": int(summ.get("n_bad_clusters", 0)),
        "notes": {
            "reject_frac": float(reject_frac),
            "margin_thresh": float(margin_thresh),
            "use_margin": bool(use_margin),
        },
        # optional: keep per-cluster dataframe if you want to print/debug
        "cluster_df": cluster_df,
        "summary": summ,
    }

    # 6) map keep_mask (indexing train_embed_loader.dataset) -> "real indices" into ds.X
    ds = train_embed_loader.dataset
    if getattr(ds, "index_map", None) is not None:
        real_indices = ds.index_map.cpu().numpy()
    else:
        real_indices = np.arange(len(ds), dtype=np.int64)

    kept_real_indices = real_indices[keep_mask]
    return kept_real_indices, report
