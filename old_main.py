# from torch.utils.data import DataLoader
# from data.dataset import UrbanSoundDataset, get_train_test_clips
# import torch
# from models.convAE_model import ConvAutoencoder
# import torch.nn.functional as F
# from tqdm import tqdm
# from data.augmentations import Compose, RandomTimeMask, RandomFreqMask, RandomGain, RandomNoise
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# train_clips, test_clips, label2id = get_train_test_clips()
#
# augment = Compose([
#     RandomTimeMask(max_width=40),
#     RandomFreqMask(max_width=20),
#     RandomGain(min_gain_db=-6, max_gain_db=6),
#     RandomNoise(noise_level=0.02),
# ])
# train_ds = UrbanSoundDataset(train_clips, label2id, transform=augment)
# test_ds = UrbanSoundDataset(test_clips, label2id)
#
# train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)
#
#
# model = ConvAutoencoder(latent_dim=128).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# # criterion = nn.MSELoss()
#
# epochs = 20
#
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for ind, (x, salience, confidence) in tqdm(enumerate(train_loader)):
#         x = x.to(device).float()
#         weights = torch.tensor(salience, dtype=torch.float32).to(device)  # optional weighting
#         optimizer.zero_grad()
#         recon, z = model(x)
#         loss = F.mse_loss(recon, x, reduction='none')  # per-element loss
#         loss = (loss * weights).mean()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * x.size(0)
#     print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader.dataset):.4f}")
#
#
# print(1)
#
# encoded = []
# categories = []
#
# with torch.no_grad():
#     for x, cat, salience, confidence in test_loader:
#         x = x.to(device).float()
#         _, z = model(x)
#         encoded.append(z.cpu())           # [batch_size, latent_dim]
#         categories.append(cat.cpu())      # [batch_size]
#
# # concatenate all batches along the sample dimension
# encoded = torch.cat(encoded, dim=0)       # shape: [N, latent_dim]
# categories = torch.cat(categories, dim=0) # shape: [N]
#
# print(encoded.shape, categories.shape)
#
# # torch.save(model.state_dict(), r"C:\data\models\audio_AE_81125aug")
# model = ConvAutoencoder(latent_dim=128).to(device)
# # model.load_state_dict(torch.load(PATH, weights_only=True))
# # model.eval()
#
#