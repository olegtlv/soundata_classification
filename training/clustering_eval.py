# training/clustering_eval.py
from sklearn.cluster import KMeans
import torch
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np


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
