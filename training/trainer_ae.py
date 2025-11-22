import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_msssim import ms_ssim  # pip install pytorch-msssim
from pytorch_msssim import ssim

def compute_combined_loss(recon, x_clean, weights=None, alpha=0.84):
    """
    recon, x_clean: [B, C, H, W] torch tensors, in range [0,1]
    weights: [B] or None  -- will be normalized to mean=1 if provided
    alpha: weight for SSIM term (0..1)
    """
    # --- ensure inputs are in [0,1] ---
    # If your x_clean is already in [0,1] skip this. Otherwise linearly scale per-sample:
    # x_min = x_clean.view(B, -1).min(dim=1)[0]; x_max = x_clean.view(B,-1).max(dim=1)[0]
    # but recommended: ensure preprocessing produces logmels scaled to [0,1] before saving/cache.

    B = recon.shape[0]

    # MS-SSIM returns mean over batch if size_average=True; we want per-sample scores.
    # pytorch-msssim supports size_average=False to get per-sample values.
    ssim_vals = ssim(
        recon, x_clean,
        data_range=1.0,
        size_average=False,
        win_size=7
    )
    # convert to "loss" form: (1 - ssim)
    ssim_loss_per_sample = 1.0 - ssim_vals  # [B]

    # per-sample MSE
    mse_per_elem = F.mse_loss(recon, x_clean, reduction='none')  # [B,C,H,W]
    mse_per_sample = mse_per_elem.view(B, -1).mean(dim=1)         # [B]

    # combined per-sample loss
    per_sample = alpha * ssim_loss_per_sample + (1.0 - alpha) * mse_per_sample

    # apply sample weights (normalize to mean = 1)
    if weights is not None:
        w = weights.to(per_sample.device).float()
        w = w / (w.mean() + 1e-9)
        per_sample = per_sample * w

    # final scalar loss
    loss = per_sample.mean()
    return loss, per_sample.mean().item(), ssim_loss_per_sample.mean().item(), mse_per_sample.mean().item()




class AETrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for batch in tqdm(loader):
            x = batch["x"].to(self.device).float()
            x_clean = batch["x_clean"].to(self.device).float()

            self.optimizer.zero_grad()

            recon, _ = self.model(x)
            weights = batch['confidence'].to(self.device)
            weights = torch.clamp(weights, min=0.5)  # min weight 0.5
            # loss = (F.mse_loss(recon, x_clean, reduction='none') * weights.view(-1, 1, 1, 1)).mean()

            # ensure x, x_clean in [0,1] (if not, scale prior to training)
            loss, avg_per_sample, avg_ssim_loss, avg_mse = compute_combined_loss(
                recon, x_clean, weights=weights, alpha=0.84
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)

        return total_loss / len(loader.dataset)

# training/trainer_vae.py

def vae_loss(recon, x_clean, mu, logvar, beta=1.0, weights=None):
    # recon, x_clean must match shape (trainer ensures match_size)
    # reconstruction loss (per-element)
    recon_elem = F.mse_loss(recon, x_clean, reduction='none')  # [B, C, H, W]

    if weights is not None:
        # weights: [B] tensor, normalize to mean=1
        w = weights.view(-1,1,1,1)
        w = w / (w.mean() + 1e-9)
        recon = (recon_elem * w).mean()
    else:
        recon = recon_elem.mean()

    # KL divergence per-batch
    # sum over latent dims, then mean over batch
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # [B]
    kl = kl.mean()

    loss = recon + beta * kl
    return loss, recon, kl


class VAETrainer:
    def __init__(self, model, optimizer, device, beta=1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.beta = beta

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            # depending on dataset dict fields
            x = batch["x"].to(self.device).float()
            x_clean = batch["x_clean"].to(self.device).float()
            # optional weights (prefer confidence or salience)
            conf = batch.get("confidence", None)
            sal = batch.get("salience", None)
            weight = None
            if conf is not None:
                weight = conf.to(self.device).float()
            elif sal is not None:
                weight = sal.to(self.device).float()

            self.optimizer.zero_grad()
            recon, mu, logvar = self.model(x)
            loss, recon_loss, kl = vae_loss(recon, x_clean, mu, logvar, beta=self.beta, weights=weight)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)

        return total_loss / len(loader.dataset)
