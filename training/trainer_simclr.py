from training.contrastive_loss import contrastive_loss
from training.trainer_ae import compute_combined_loss
import torch.nn.functional as F
from tqdm import tqdm
import torch


def batch_entropy_regularizer(logits, eps=1e-6):
    p = F.softmax(logits, dim=1)      # [B,K]
    m = p.mean(dim=0)                 # [K]
    return (m * (m + eps).log()).sum()  # negative entropy (minimize -> maximize entropy)

@torch.no_grad()
def sharpen_probs(logits, temp=0.1):
    # logits: [B,K]
    p = F.softmax(logits / temp, dim=1)
    return p

def proto_ce_loss(q, logits):
    # q: [B,K] target probs, logits: [B,K]
    logp = F.log_softmax(logits, dim=1)
    return -(q * logp).sum(dim=1).mean()

def swav_lite_loss(logits1, logits2, t_assign=0.1):
    # stop-grad targets from opposite view
    q1 = sharpen_probs(logits1.detach(), temp=t_assign)
    q2 = sharpen_probs(logits2.detach(), temp=t_assign)

    # swapped prediction
    return proto_ce_loss(q1, logits2) + proto_ce_loss(q2, logits1)


def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def var_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(1.0 - std).mean()

def cov_loss(z: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(dim=0)
    b = z.shape[0]
    c = (z.T @ z) / (b - 1)
    return off_diagonal(c).pow(2).mean()

class SimCLRrainer:
    def __init__(self, model, optimizer, device, temperature=0.05, lambda_recon=0., amp=True, grad_clip=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.temperature = temperature
        self.lambda_recon = lambda_recon
        self.amp = amp
        self.grad_clip = grad_clip
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0

        if epoch < 1:
            freeze(self.model.encoder)
        else:
            unfreeze(self.model.encoder)

        for batch in tqdm(loader):
            x1 = batch["x1"].to(self.device).float()
            x2 = batch["x2"].to(self.device).float()
            x1 = x1.to(memory_format=torch.channels_last)
            x2 = x2.to(memory_format=torch.channels_last)
            x_clean = batch["x_clean"].to(self.device).float() if "x_clean" in batch else None

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.amp):
                # raw encoder features (good for var/cov)
                z1 = self.model.encoder.encode(x1)
                z2 = self.model.encoder.encode(x2)

                # projector for contrastive loss
                p1 = self.model.projector(z1)
                p2 = self.model.projector(z2)


                sal_w = batch.get("sal_w", None)
                if sal_w is not None:
                    sal_w = sal_w.to(self.device).float().view(-1)
                    if sal_w.numel() != x1.size(0):
                        raise RuntimeError(f"sal_w has {sal_w.numel()} elems, batch has {x1.size(0)}")

                # optionally combine with confidence if you want (your confidence is all-ones currently)
                conf_w = batch.get("confidence", None)
                if conf_w is not None:
                    conf_w = conf_w.to(self.device).float().view(-1)
                else:
                    conf_w = None

                dyn_w = batch.get("dyn_w", None)
                if dyn_w is not None:
                    dyn_w = dyn_w.to(self.device).float().view(-1)
                else:
                    dyn_w = None

                # existing: sal_w, conf_w -> weights
                weights = None
                for w in [sal_w, conf_w, dyn_w]:
                    if w is None:
                        continue
                    weights = w if weights is None else (weights * w)

                con_loss = contrastive_loss(p1, p2, weights=weights, temperature=self.temperature)

                # con_loss = contrastive_loss(p1, p2, weights=None, temperature=self.temperature)

                # VICReg-style regularizers on raw z
                loss_var = var_loss(z1) + var_loss(z2)
                loss_cov = cov_loss(z1) + cov_loss(z2)

                # NOTE: recon loss is incompatible with ResNetEncoder unless you have a decoder.
                # keep it off (lambda_recon=0) for ResNet SimCLR.
                if x_clean is not None and self.lambda_recon > 0:
                    recon, _ = self.model.encoder(x_clean)
                    weights = torch.clamp(batch['confidence'].to(self.device), min=0.5)
                    recon_loss, _, _, _ = compute_combined_loss(recon, x_clean, weights=weights, alpha=0.9)
                else:
                    recon_loss = 0.0

                lam_var = 0.75
                lam_cov = 0.075

                loss = con_loss + self.lambda_recon * recon_loss + lam_var * loss_var + lam_cov * loss_cov
                # projector outputs p1,p2: [B,D]
                # l1 = self.model.prototypes(p1)  # [B,K]
                # l2 = self.model.prototypes(p2)
                #
                # loss_proto = swav_lite_loss(l1, l2, t_assign=0.1)
                # loss_bal = batch_entropy_regularizer(l1) + batch_entropy_regularizer(l2)

                # loss = loss_proto + 0.1 * loss_bal + lam_var * loss_var + lam_cov * loss_cov

            # AMP-safe backward/step
            self.scaler.scale(loss).backward()

            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * x1.size(0)

        return total_loss / len(loader.dataset)
