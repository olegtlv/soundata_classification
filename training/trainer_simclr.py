from training.contrastive_loss import contrastive_loss
from training.trainer_ae import compute_combined_loss
import torch.nn.functional as F
from tqdm import tqdm
import torch
def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

class SimCLRrainer:
    def __init__(self, model, optimizer, device, temperature=0.5, lambda_recon=0.1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.temperature = temperature
        self.lambda_recon = lambda_recon

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0

        if epoch < 10:
            freeze(self.model.encoder)
        else:
            unfreeze(self.model.encoder)

        for batch in tqdm(loader):
            x1 = batch["x1"].to(self.device).float()
            x2 = batch["x2"].to(self.device).float()
            x_clean = batch["x"].to(self.device).float() if "x" in batch else None

            self.optimizer.zero_grad()

            p1 = self.model(x1)      # [B, D]
            p2 = self.model(x2)      # [B, D]

            weights = batch['confidence'].to(self.device)
            weights = torch.clamp(weights, min=0.5)  # min weight 0.5
            # loss = (F.mse_loss(recon, x_clean, reduction='none') * weights.view(-1, 1, 1, 1)).mean()

            # ensure x, x_clean in [0,1] (if not, scale prior to training)
            con_loss  = contrastive_loss(p1, p2, weights=None, temperature=self.temperature)
            # reconstruction loss
            if x_clean is not None:
                recon, _ = self.model.encoder(x_clean)   # AE forward
                recon_loss, _, _, _ = compute_combined_loss(
                    recon, x_clean, weights=weights, alpha=0.9)
            else:
                recon_loss = 0.0
            #
            # std_z = torch.std(torch.cat([p1, p2], dim=0), dim=0)
            # var_loss = torch.mean(F.relu(1 - std_z))
            loss = con_loss + self.lambda_recon * recon_loss #+ 0.04 * var_loss

            # loss = con_loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x1.size(0)

        return total_loss / len(loader.dataset)