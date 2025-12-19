import torch
from tqdm import tqdm
from models.byol_model import byol_loss


class BYOLTrainer:
    def __init__(self, model, optimizer, device, total_epochs, ema_base=0.996):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.ema_base = ema_base
        self.total_epochs = total_epochs

    def train_epoch(self, loader, epoch, ):
        self.model.train()
        total_loss = 0.0

        # optional: cosine schedule for EMA momentum (common in BYOL)
        # m goes from ema_base -> 1.0 over training
        def ema_momentum(e):
            if self.total_epochs <= 1:
                return self.ema_base
            t = e / (self.total_epochs - 1)
            return 1.0 - (1.0 - self.ema_base) * (0.5 * (1.0 + torch.cos(torch.tensor(t * 3.1415926535))).item())

        m = ema_momentum(epoch)

        for batch in tqdm(loader, desc=f"BYOL epoch {epoch}"):
            x1 = batch["x1"].to(self.device).float()  # [B, 1, F, T]
            x2 = batch["x2"].to(self.device).float()

            # online forward
            _, p1, q1 = self.model.forward_online(x1)
            _, p2, q2 = self.model.forward_online(x2)

            # target forward (no grad)
            with torch.no_grad():
                _, t1 = self.model.forward_target(x1)  # target projection of view1
                _, t2 = self.model.forward_target(x2)

            # symmetric BYOL loss
            loss = byol_loss(q1, t2) + byol_loss(q2, t1)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # EMA target update
            self.model.update_target(m=m)

            total_loss += loss.item()

        return total_loss / max(1, len(loader))
