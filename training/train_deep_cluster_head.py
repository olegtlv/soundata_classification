import torch.nn.functional as F

def train_deep_cluster_head(model, loader, optimizer, criterion, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        lambda_recon = 0.33
        for x, pseudo in loader:
            x = x.to(device)          # [B, 1, F, T]
            pseudo = pseudo.to(device)
            # recon, z = model.encoder(x)

            optimizer.zero_grad()
            logits = model(x)         # [B, K]
            ce_loss = criterion(logits, pseudo)
            # recon_loss = F.mse_loss(recon, x)
            loss = ce_loss #+ lambda_recon * recon_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        print(f"Epoch {epoch+1}, loss = {total_loss / len(loader.dataset):.4f}")

