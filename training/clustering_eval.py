# training/clustering_eval.py
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import torch
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
import numpy as np


def evaluate_clustering(Z, labels, n_clusters):
    km = KMeans(n_clusters=n_clusters)
    preds = km.fit_predict(Z)

    nmi = normalized_mutual_info_score(labels, preds)
    ari = adjusted_rand_score(labels, preds)
    return nmi, ari, preds



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
