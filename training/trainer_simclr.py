from training.contrastive_loss import contrastive_loss
import torch.nn.functional as F
from tqdm import tqdm

class SimCLRrainer:
    def __init__(self, model, optimizer, device, temperature=0.5, lambda_recon=0.1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.temperature = temperature
        self.lambda_recon = lambda_recon

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for batch in tqdm(loader):
            x1 = batch["x1"].to(self.device).float()
            x2 = batch["x2"].to(self.device).float()
            x_clean = batch["x"].to(self.device).float() if "x" in batch else None

            self.optimizer.zero_grad()

            p1 = self.model(x1)      # [B, D]
            p2 = self.model(x2)      # [B, D]

            # weights = batch['confidence'].to(self.device)
            # weights = torch.clamp(weights, min=0.5)  # min weight 0.5
            # loss = (F.mse_loss(recon, x_clean, reduction='none') * weights.view(-1, 1, 1, 1)).mean()

            # ensure x, x_clean in [0,1] (if not, scale prior to training)
            con_loss  = contrastive_loss(p1,p2, weights=None, temperature=self.temperature)
            # reconstruction loss
            if x_clean is not None:
                recon, _ = self.model.encoder(x_clean)   # AE forward
                recon_loss = F.mse_loss(recon, x_clean)
            else:
                recon_loss = 0.0

            loss = con_loss + self.lambda_recon * recon_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x1.size(0)

        return total_loss / len(loader.dataset)