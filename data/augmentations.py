
import torch
import random

def get_ae_augment():
    return Compose([
        RandomTimeMask(max_width=20),
        RandomFreqMask(max_width=10),
        RandomGain(-3, 3),
        RandomNoise(0.03),
        EnergyBiasedRandomCrop(128),
        RandomTimeShift(max_shift=10),
        RandomFreqShift(max_shift=2)
    ])

class RandomTimeMask:
    def __init__(self, max_width=40):
        self.max_width = max_width

    def __call__(self, spec):
        _, F, T = spec.shape
        w = random.randint(0, self.max_width)
        if w == 0:
            return spec
        t = random.randint(0, max(0, T - w))

        out = spec.clone()
        # per-frequency mean over time: [1, F, 1]
        fill = spec.mean(dim=2, keepdim=True)
        # broadcast to [1, F, w]
        out[:, :, t:t+w] = fill.expand(-1, -1, w)
        return out


class RandomFreqMask:
    def __init__(self, max_width=20):
        self.max_width = max_width

    def __call__(self, spec):
        _, F, T = spec.shape
        w = random.randint(0, self.max_width)
        if w == 0:
            return spec
        f = random.randint(0, max(0, F - w))

        out = spec.clone()
        # per-frequency mean over time: [1, F, 1]
        fill = spec.mean(dim=2, keepdim=True)          # [1, F, 1]
        fill_band = fill[:, f:f+w, :]                  # [1, w, 1]
        out[:, f:f+w, :] = fill_band.expand(-1, -1, T) # [1, w, T]
        return out



class RandomGain:
    def __init__(self, min_gain_db=-6, max_gain_db=6):
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def __call__(self, spec: torch.Tensor):
        gain = random.uniform(self.min_gain_db, self.max_gain_db)
        factor = 10 ** (gain / 20)
        return spec * factor


class RandomNoise:
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level

    def __call__(self, spec: torch.Tensor):
        noise = torch.randn_like(spec) * self.noise_level
        return spec + noise

# class RandomTimeReverse:
#     def __call__(self, x):
#         if random.random() < 0.5:
#             return torch.flip(x, dims=[2])  # flip time dimension
#         return x
class MultiTimeMask:
    def __init__(self, max_width=10, n_masks=(1, 3)):
        self.max_width = max_width
        self.n_masks = n_masks
    def __call__(self, x):
        _, F, T = x.shape
        out = x.clone()
        k = random.randint(self.n_masks[0], self.n_masks[1])
        fill = out.mean()
        for _ in range(k):
            w = random.randint(0, self.max_width)
            if w == 0:
                continue
            t = random.randint(0, max(0, T - w))
            out[:, :, t:t+w] = fill
        return out


class BackgroundPool:
    def __init__(self, X_tensor, max_items=2048):
        # X_tensor: [N, 1, F, T]
        self.X = X_tensor
        self.max_items = min(max_items, len(X_tensor))

    def sample(self):
        i = random.randint(0, self.max_items - 1)
        return self.X[i]

class MixBackground:
    def __init__(self, bg_sampler=None, alpha=(0.02, 0.15), p=0.5):
        self.bg_sampler = bg_sampler
        self.alpha = alpha
        self.p = p

    def __call__(self, x: torch.Tensor, x_bg: torch.Tensor | None = None):
        if random.random() > self.p:
            return x

        if x_bg is None:
            if self.bg_sampler is None:
                return x
            x_bg = self.bg_sampler.sample()

        x_bg = x_bg.to(x.device)

        # x: [1,F,T], x_bg: [1,F,T] or [1,F,Tb]
        if x_bg.shape[-1] != x.shape[-1]:
            T = x.shape[-1]
            Tb = x_bg.shape[-1]
            if Tb > T:
                start = random.randint(0, Tb - T)
                x_bg = x_bg[:, :, start:start+T]
            else:
                x_bg = torch.nn.functional.pad(x_bg, (0, T - Tb))

        a = random.uniform(*self.alpha)
        return x + a * x_bg



class RandomGaussianNoise:
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x):
        return x + torch.randn_like(x) * self.std

class RandomCrop:
    def __init__(self, target_frames):
        self.target_frames = target_frames

    def __call__(self, x):
        _, _, T = x.shape
        if T <= self.target_frames:
            return x
        start = random.randint(0, T - self.target_frames)
        return x[:, :, start:start + self.target_frames]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class ContrastiveAug:
    """Applies a random chain of augmentations for contrastive learning."""
    def __init__(self, target_frames=128):
        self.augmentations = [
            RandomFreqMask(12),
            RandomTimeMask(20),
            RandomGaussianNoise(0.01),
            # RandomTimeReverse(),
        ]
        self.crop = RandomCrop(target_frames)

    def __call__(self, x):
        x = self.crop(x)

        # random order, random subset
        augs = random.sample(self.augmentations, k=random.randint(1, len(self.augmentations)))

        for aug in augs:
            x = aug(x)

        return x

class SafeContrastiveAug:
    def __call__(self, x):
        # 1. Mild gaussian noise
        if random.random() < 0.5:
            x = x + 0.01 * torch.randn_like(x)

        # 2. Mild gain
        if random.random() < 0.5:
            gain_db = random.uniform(-3, 3)
            factor = 10 ** (gain_db / 20)
            x = x * factor

        # 3. Small time mask (tiny)
        if random.random() < 0.3:
            width = random.randint(0, 5)
            t = random.randint(0, x.shape[-1] - width)
            x[:, :, t:t+width] = 0

        # 4. Small freq mask (tiny)
        if random.random() < 0.3:
            width = random.randint(0, 3)
            f = random.randint(0, x.shape[-2] - width)
            x[:, f:f+width, :] = 0

        return x


class RandomTimeShift:
    def __init__(self, max_shift=12, pad_value=0.0):
        self.max_shift = int(max_shift)
        self.pad_value = pad_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, F, T]
        if self.max_shift <= 0:
            return x
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x

        _, F, T = x.shape
        out = x.new_full((1, F, T), self.pad_value)

        if shift > 0:
            out[:, :, shift:] = x[:, :, : T - shift]
        else:
            s = -shift
            out[:, :, : T - s] = x[:, :, s:]
        return out


class RandomFreqShift:
    def __init__(self, max_shift=2, pad_value=0.0):
        self.max_shift = int(max_shift)
        self.pad_value = pad_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, F, T]
        if self.max_shift <= 0:
            return x
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x

        _, F, T = x.shape
        out = x.new_full((1, F, T), self.pad_value)

        if shift > 0:
            out[:, shift:, :] = x[:, : F - shift, :]
        else:
            s = -shift
            out[:, : F - s, :] = x[:, s:, :]
        return out

class EnergyBiasedRandomCrop:
    def __init__(self, target_frames=128, bias_prob=0.7):
        self.target_frames = target_frames
        self.bias_prob = bias_prob

    def __call__(self, x):
        # x: [1, F, T]
        _, F, T = x.shape
        if T <= self.target_frames:
            return x

        if random.random() < self.bias_prob:
            # energy over time
            energy = x.mean(dim=1).squeeze(0)  # [T]
            energy = torch.nn.functional.avg_pool1d(
                energy[None, None, :],
                kernel_size=5,
                stride=1,
                padding=2,
            ).squeeze()

            # sample center with probability proportional to energy
            probs = energy / (energy.sum() + 1e-6)
            center = torch.multinomial(probs, 1).item()
            start = max(0, min(center - self.target_frames // 2, T - self.target_frames))
        else:
            start = random.randint(0, T - self.target_frames)

        return x[:, :, start : start + self.target_frames]

class TransientBiasedRandomCrop:
    def __init__(self, target_frames=128, bias_prob=0.95, smooth=5):
        self.target_frames = target_frames
        self.bias_prob = bias_prob
        self.smooth = smooth

    def __call__(self, x):
        # x: [1, F, T]
        _, F, T = x.shape
        if T <= self.target_frames:
            return x

        if random.random() < self.bias_prob:
            # spectral flux-ish: positive frame-to-frame change
            s = x.squeeze(0)  # [F,T]
            diff = torch.relu(s[:, 1:] - s[:, :-1])  # [F, T-1]
            score = diff.mean(dim=0)  # [T-1]
            # pad to length T
            score = torch.cat([score[:1], score], dim=0)  # [T]

            # smooth
            score = torch.nn.functional.avg_pool1d(
                score[None, None, :], kernel_size=self.smooth, stride=1, padding=self.smooth // 2
            ).squeeze()

            probs = score / (score.sum() + 1e-6)
            center = torch.multinomial(probs, 1).item()
            start = max(0, min(center - self.target_frames // 2, T - self.target_frames))
        else:
            start = random.randint(0, T - self.target_frames)

        return x[:, :, start:start + self.target_frames]



class StrongContrastiveAug:
    def __init__(self, target_frames=128):
        self.crop = EnergyBiasedRandomCrop(target_frames)
        self.tshift = RandomTimeShift(max_shift=12)
        self.fshift = RandomFreqShift(max_shift=2)
        self.time_mask = RandomTimeMask(max_width=8)
        # self.freq_mask = RandomFreqMask(max_width=10)
        self.gain = RandomGain(-6, 6)
        self.noise = RandomNoise(0.02)

    def __call__(self, x):
        x = self.crop(x)

        if random.random() < 0.8:
            x = self.tshift(x)

        if random.random() < 0.3:
            x = self.fshift(x)

        if random.random() < 0.3:
            x = self.time_mask(x)

        if random.random() < 0.9:
            x = self.freq_mask(x)

        if random.random() < 0.5:
            x = self.gain(x)

        if random.random() < 0.7:
            x = self.noise(x)

        return x


from dataclasses import dataclass
import random

@dataclass
class AugConfig:
    target_frames: int = 128

    p_tshift: float = 0.8
    p_fshift: float = 0.2
    p_time_mask: float = 0.2
    p_freq_mask: float = 0.0
    p_gain: float = 0.5
    p_noise: float = 0.5

    # strengths
    max_tshift: int = 12
    max_fshift: int = 2
    time_mask_w: int = 8
    freq_mask_w: int = 10
    gain_db: tuple = (-6, 6)
    noise_std: float = 0.02

def make_aug_config(mode: str, target_frames=128) -> AugConfig:
    """
    mode:
      - "simclr" / "byol": two-view contrastive augs
      - "ae": minimal augs (reconstruction)
      - "dae": denoising AE (aug input, clean target)
    """
    if mode in ("simclr", "byol"):
        # safer than your current StrongContrastiveAug (mainly freq_mask)
        return AugConfig(
            target_frames=target_frames,
            p_tshift=0.8,
            p_fshift=0.05,
            p_time_mask=0.25,
            p_freq_mask=0.05,      # << was 0.9
            p_gain=0.5,
            p_noise=0.5,
            max_tshift=12,
            max_fshift=1,
            time_mask_w=8,
            freq_mask_w=6,         # << was 10
            gain_db=(-6, 6),
            noise_std=0.02,
        )

    if mode == "ae":
        # reconstruction wants stability
        return AugConfig(
            target_frames=target_frames,
            p_tshift=0.0,
            p_fshift=0.0,
            p_time_mask=0.0,
            p_freq_mask=0.0,
            p_gain=0.2,
            p_noise=0.2,
            gain_db=(-3, 3),
            noise_std=0.01,
        )

    if mode == "dae":
        # denoising AE: allow noise/gain + mild masks (optional)
        return AugConfig(
            target_frames=target_frames,
            p_tshift=0.1,
            p_fshift=0.0,
            p_time_mask=0.1,
            p_freq_mask=0.1,
            p_gain=0.4,
            p_noise=0.5,
            time_mask_w=6,
            freq_mask_w=4,
            gain_db=(-6, 6),
            noise_std=0.02,
        )

    raise ValueError(f"Unknown mode={mode}")


class UnifiedAug:
    """
    Uses your exact ops:
      EnergyBiasedRandomCrop, RandomTimeShift, RandomFreqShift,
      RandomTimeMask, RandomFreqMask, RandomGain, RandomNoise
    """
    def __init__(self, cfg: AugConfig):
        self.cfg = cfg
        # self.crop = EnergyBiasedRandomCrop(cfg.target_frames)
        self.crop = TransientBiasedRandomCrop(cfg.target_frames)
        self.tshift = RandomTimeShift(max_shift=cfg.max_tshift)
        self.fshift = RandomFreqShift(max_shift=cfg.max_fshift)
        self.time_mask = RandomTimeMask(max_width=cfg.time_mask_w)
        self.freq_mask = RandomFreqMask(max_width=cfg.freq_mask_w)
        self.gain = RandomGain(cfg.gain_db[0], cfg.gain_db[1])
        self.noise = RandomNoise(cfg.noise_std)

    def __call__(self, x):
        # x: [1, F, T] or [C, F, T] depending on your dataset
        x = self.crop(x)

        if random.random() < self.cfg.p_tshift:
            x = self.tshift(x)
        if random.random() < self.cfg.p_fshift:
            x = self.fshift(x)
        if random.random() < self.cfg.p_time_mask:
            x = self.time_mask(x)
        if random.random() < self.cfg.p_freq_mask:
            x = self.freq_mask(x)
        if random.random() < self.cfg.p_gain:
            x = self.gain(x)
        if random.random() < self.cfg.p_noise:
            x = self.noise(x)

        return x

cfg_mild = AugConfig(
    target_frames=128,
    p_tshift=0.6, max_tshift=4,
    p_fshift=0.02, max_fshift=1,
    p_time_mask=0.1, time_mask_w=3,
    p_freq_mask=0.02, freq_mask_w=2,
    p_gain=0.2, gain_db=(-3, 3),
    p_noise=0.3, noise_std=0.01,
)

cfg_strong = AugConfig(
    target_frames=128,
    p_tshift=0.8, max_tshift=12,
    p_fshift=0.1, max_fshift=2,
    p_time_mask=0.3, time_mask_w=10,
    p_freq_mask=0.08, freq_mask_w=6,
    p_gain=0.6, gain_db=(-6, 6),
    p_noise=0.5, noise_std=0.02,
)

import random
import torch

class UnifiedPostCropAug:
    """
    Same ops as UnifiedAug, but assumes x is already cropped.
    Applies shifts/masks/gain/noise *after* cropping.
    """
    def __init__(self, cfg: AugConfig):
        self.cfg = cfg
        self.tshift = RandomTimeShift(max_shift=cfg.max_tshift)
        self.fshift = RandomFreqShift(max_shift=cfg.max_fshift)
        self.time_mask = RandomTimeMask(max_width=cfg.time_mask_w)
        self.freq_mask = RandomFreqMask(max_width=cfg.freq_mask_w)
        self.gain = RandomGain(cfg.gain_db[0], cfg.gain_db[1])
        self.noise = RandomNoise(cfg.noise_std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is already [1, F, T] cropped to target_frames
        if random.random() < self.cfg.p_tshift:
            x = self.tshift(x)
        if random.random() < self.cfg.p_fshift:
            x = self.fshift(x)
        if random.random() < self.cfg.p_time_mask:
            x = self.time_mask(x)
        if random.random() < self.cfg.p_freq_mask:
            x = self.freq_mask(x)
        if random.random() < self.cfg.p_gain:
            x = self.gain(x)
        if random.random() < self.cfg.p_noise:
            x = self.noise(x)
        return x

def unsupervised_salience_weight(
    spec: torch.Tensor,
    w_min: float = 0.10,
    w_max: float = 1.00,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    spec: [1,F,T] log-mel after crop.
    Returns scalar weight in [w_min,w_max].
    Combines:
      - transient peakiness over time
      - spectral flatness penalty (noise-like -> lower weight)
    """
    x = spec.float().squeeze(0)  # [F,T]

    # ---- 1) transient peakiness over time ----
    e_t = x.mean(dim=0)                      # [T]
    med = e_t.median()
    p90 = e_t.quantile(0.90)
    # robust "how much above typical"
    peakiness = (p90 - med)                  # log domain diff

    # ---- 2) spectral flatness / entropy (penalize flat noise) ----
    # use average spectrum over time; convert to positive "power-ish"
    s_f = x.mean(dim=1)                      # [F]
    p = torch.softmax(s_f, dim=0)            # [F], sums to 1
    entropy = -(p * (p + eps).log()).sum()   # [0, log(F)]
    entropy_norm = entropy / (torch.log(torch.tensor(float(p.numel()), device=x.device)) + eps)
    # entropy_norm close to 1 => flat/noisy; close to 0 => peaky/structured
    structure = 1.0 - entropy_norm

    # ---- combine and map to weight ----
    # peakiness is in log units; clamp to a reasonable range
    peakiness = torch.clamp(peakiness, -2.0, 6.0)
    # scale to 0..1
    peak01 = torch.sigmoid(peakiness)        # 0..1

    score01 = 0.6 * peak01 + 0.4 * structure
    w = w_min + (w_max - w_min) * score01
    return w


class TwoViewAug:
    """
    Shared crop (event-aligned) + asymmetric post-crop corruption.
    Optionally: mix background using x_bg.
    Returns: x1, x2, sal_w
    """
    def __init__(self, cfg_mild, cfg_strong, shared_crop="transient", mixbg=None,
                 sal_w_min=0.10, sal_w_max=1.00):
        self.cfg_mild = cfg_mild
        self.cfg_strong = cfg_strong
        self.mixbg = mixbg
        self.sal_w_min = sal_w_min
        self.sal_w_max = sal_w_max

        if shared_crop == "transient":
            self.crop = TransientBiasedRandomCrop(target_frames=cfg_mild.target_frames, bias_prob=0.85, smooth=9)
        elif shared_crop == "energy":
            self.crop = EnergyBiasedRandomCrop(target_frames=cfg_mild.target_frames, bias_prob=0.7)
        elif shared_crop == "random":
            self.crop = RandomCrop(cfg_mild.target_frames)
        else:
            raise ValueError(f"shared_crop must be one of: transient/energy/random, got {shared_crop}")

        self.aug_mild = UnifiedPostCropAug(cfg_mild)
        self.aug_strong = UnifiedPostCropAug(cfg_strong)

    def __call__(self, x: torch.Tensor, x_bg: torch.Tensor | None = None):
        # 1) event-aligned "instance"
        x0 = self.crop(x)

        # compute salience weight on the cropped instance (before heavy corruption)
        sal_w = unsupervised_salience_weight(x0, w_min=self.sal_w_min, w_max=self.sal_w_max)

        # 2) mild view from same crop
        x1 = self.aug_mild(x0.clone())

        # 3) strong view from same crop
        x2 = self.aug_strong(x0.clone())

        # 4) optional background mixing only if x_bg is provided
        if self.mixbg is not None and x_bg is not None:
            # (you decide what MixBackground expects; common: mix in background-ish spec)
            x2 = self.mixbg(x2, x_bg)

        return x1, x2, sal_w
