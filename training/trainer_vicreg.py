# training/trainer_vicreg.py
import torch
from tqdm import tqdm
from training.vicreg_loss import vicreg_loss


class VICRegTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        sim_coeff=25.0,
        var_coeff=25.0,
        cov_coeff=1.0,
        grad_clip=None,
        amp=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.grad_clip = grad_clip
        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def train_epoch(self, loader, epoch: int):
        self.model.train()
        total = 0.0
        n = 0

        pbar = tqdm(loader, desc=f"VICReg epoch {epoch}")
        for batch in pbar:
            x1 = batch["x1"].to(self.device).float()
            x2 = batch["x2"].to(self.device).float()

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.amp):
                _, y1 = self.model(x1)
                _, y2 = self.model(x2)
                loss, stats = vicreg_loss(
                    y1, y2,
                    sim_coeff=self.sim_coeff,
                    var_coeff=self.var_coeff,
                    cov_coeff=self.cov_coeff,
                )

            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total += loss.item()
            n += 1
            pbar.set_postfix(
                loss=float(stats["loss"]),
                sim=float(stats["sim"]),
                var=float(stats["var"]),
                cov=float(stats["cov"]),
            )

        return total / max(1, n)
