import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_msssim import ms_ssim  # pip install pytorch-msssim
from pytorch_msssim import ssim
from training.contrastive_loss import contrastive_loss


def _match_size(a, b):
    """
    Center-crop tensors a and b to the same (H,W).
    a, b: [B,C,H,W]
    Returns: (a_cropped, b_cropped)
    """
    B, C, H1, W1 = a.shape
    _, _, H2, W2 = b.shape
    H = min(H1, H2)
    W = min(W1, W2)

    # crop starts (center crop)
    h1s = (H1 - H) // 2
    w1s = (W1 - W) // 2
    h2s = (H2 - H) // 2
    w2s = (W2 - W) // 2

    a = a[:, :, h1s:h1s+H, w1s:w1s+W]
    b = b[:, :, h2s:h2s+H, w2s:w2s+W]
    return a, b


def compute_combined_loss(recon, x_clean, weights=None, alpha=0.84):
    """
    recon, x_clean: [B,C,H,W] tensors in range [0,1].
    alpha: weight for SSIM term.
    weights: optional per-sample confidence weights.

    Returns:
        loss_scalar, avg_loss, avg_ssim_loss, avg_mse_loss
    """

    # ---------------------------------------------------
    # 1. Crop recon + x_clean to the same size (SSIM requirement)
    # ---------------------------------------------------
    recon, x_clean = _match_size(recon, x_clean)

    B = recon.size(0)

    # ---------------------------------------------------
    # 2. SSIM loss (per sample)
    # ---------------------------------------------------
    ssim_vals = ssim(
        recon, x_clean,
        data_range=1.0,
        size_average=False,  # get [B]
        win_size=7
    )
    ssim_loss_per_sample = 1.0 - ssim_vals  # [B]

    # ---------------------------------------------------
    # 3. MSE loss (per sample)
    # ---------------------------------------------------
    mse_vals = F.mse_loss(recon, x_clean, reduction="none")   # [B,C,H,W]
    mse_loss_per_sample = mse_vals.view(B, -1).mean(dim=1)    # [B]

    # ---------------------------------------------------
    # 4. Combine losses
    # ---------------------------------------------------
    per_sample = alpha * ssim_loss_per_sample + (1 - alpha) * mse_loss_per_sample

    # ---------------------------------------------------
    # 5. Optional per-sample weights (normalized to mean=1)
    # ---------------------------------------------------
    if weights is not None:
        w = weights.to(per_sample.device).float()
        w = w / (w.mean() + 1e-9)
        per_sample = per_sample * w

    # ---------------------------------------------------
    # 6. Final scalar
    # ---------------------------------------------------
    loss = per_sample.mean()

    return (
        loss,
        per_sample.mean().item(),
        ssim_loss_per_sample.mean().item(),
        mse_loss_per_sample.mean().item()
    )

class AETrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    # def train_with_contrastive(self, loader):
    #     self.model.train()
    #     total_loss = 0
    #
    #     for batch in loader:
    #         x1 = batch["x1"].to(self.device)
    #         x2 = batch["x2"].to(self.device)
    #         clean = batch["x"].to(self.device)
    #
    #         # --- AE reconstruction ---
    #         recon, z1 = self.model(x1)
    #         loss_recon, avg_total, avg_ssim, avg_mse = compute_combined_loss(recon, clean)
    #
    #         # --- second view encoding (no decode needed) ---
    #         with torch.no_grad():
    #             _, z2 = self.model(x2)
    #
    #         # --- weak contrastive ---
    #         loss_contrast = contrastive_loss(z1, z2)
    #
    #         # --- combined ---
    #         loss = loss_recon + 0.05 * loss_contrast
    #
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #
    #     return total_loss / len(loader.dataset)

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0

        for batch in tqdm(loader):
            x = batch["x1"].to(self.device).float()
            x2 = batch["x2"].to(self.device)

            x_clean = batch["x_clean"].to(self.device).float()

            self.optimizer.zero_grad()

            recon, z1 = self.model(x)
            _, z2 = self.model(x2)

            loss_contrast = contrastive_loss(z1, z2)

            weights = batch['confidence'].to(self.device)
            weights = torch.clamp(weights, min=0.5)  # min weight 0.5
            # loss = (F.mse_loss(recon, x_clean, reduction='none') * weights.view(-1, 1, 1, 1)).mean()

            # ensure x, x_clean in [0,1] (if not, scale prior to training)
            loss, avg_per_sample, avg_ssim_loss, avg_mse = compute_combined_loss(
                recon, x_clean, weights=weights, alpha=0.84
            )
            loss = loss + 0.05 * loss_contrast

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
            x_clean = batch["x"].to(self.device).float()
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
