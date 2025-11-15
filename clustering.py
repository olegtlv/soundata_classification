from models.convAE_model import ConvAutoencoder
import torch
from data.dataset import UrbanSoundDataset
import soundata
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import plotly.express as px
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# {'air_conditioner': 0,
#  'car_horn': 1,
#  'children_playing': 2,
#  'dog_bark': 3,
#  'drilling': 4,
#  'engine_idling': 5,
#  'gun_shot': 6,
#  'jackhammer': 7,
#  'siren': 8,
#  'street_music': 9}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ConvAutoencoder(latent_dim=128).to(device)
model.load_state_dict(torch.load(r"C:\data\models\audio_AE_141125aug_norm", weights_only=True))
model.eval()

dataset = soundata.initialize('urbansound8k')
clips = [dataset.clip(cid) for cid in dataset.clip_ids]
classes = sorted(set(c.class_label for c in clips))
label2id = {c: i for i, c in enumerate(classes)}

test_clips = [c for c in clips if c.fold == 10]
test_ds = UrbanSoundDataset(test_clips, label2id)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)


encoded = []
categories = []

with torch.no_grad():
    for samp in test_loader:
        x = samp['x_clean'].to(device).float()
        _, z = model(x)
        encoded.append(z.cpu())           # [batch_size, latent_dim]
        categories.append(samp['label'].cpu())      # [batch_size]

# concatenate all batches along the sample dimension
encoded = torch.cat(encoded, dim=0)       # shape: [N, latent_dim]
categories = torch.cat(categories, dim=0) # shape: [N]

print(encoded.shape, categories.shape)


# Convert to numpy for sklearn
X = encoded.numpy()
y = categories.numpy()

# Choose number of clusters (e.g. same as #classes)
n_clusters = len(np.unique(y))
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X)


# t-SNE for 2D visualization
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_2d = tsne.fit_transform(X)



# --- Create DataFrame for Plotly
import pandas as pd
df = pd.DataFrame({
    'x': X_2d[:, 0],
    'y': X_2d[:, 1],
    'category': y
})

# --- Plot with Plotly Express
fig = px.scatter(
    df,
    x='x', y='y',
    color=df['category'].astype(str),
    title="t-SNE visualization of encoded embeddings (colored by category)",
    opacity=0.7,
    template="plotly_dark"
)


k = len(np.unique(y))
kmeans = KMeans(n_clusters=k).fit(encoded)

nmi = normalized_mutual_info_score(y, kmeans.labels_)
ari = adjusted_rand_score(y, kmeans.labels_)

print(f"NMI: {nmi:.3f}, ARI: {ari:.3f}")


# --- Save interactive HTML
output_path = r"C:\tmp\encoded_tsne.html"
fig.write_html(output_path)

print(f"✅ Saved interactive visualization to: {output_path}")

# Optional: display in notebook
# fig.show()