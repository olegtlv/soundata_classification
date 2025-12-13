
import torch
import random

def get_ae_augment():
    return Compose([
        RandomTimeMask(max_width=20),
        RandomFreqMask(max_width=10),
        RandomGain(-3, 3),
        RandomNoise(0.02),
    ])


class RandomTimeMask:
    def __init__(self, max_width=40):
        self.max_width = max_width

    def __call__(self, spec: torch.Tensor):
        # spec shape: [1, freq, time] or [freq, time]
        _, freq, time = spec.shape
        w = random.randint(0, self.max_width)
        t = random.randint(0, max(0, time - w))
        spec = spec.clone()
        spec[:, :, t:t+w] = 0
        return spec


class RandomFreqMask:
    def __init__(self, max_width=20):
        self.max_width = max_width

    def __call__(self, spec: torch.Tensor):
        _, freq, time = spec.shape
        w = random.randint(0, self.max_width)
        f = random.randint(0, max(0, freq - w))
        spec = spec.clone()
        spec[:, f:f+w, :] = 0
        return spec


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

class RandomTimeReverse:
    def __call__(self, x):
        if random.random() < 0.5:
            return torch.flip(x, dims=[2])  # flip time dimension
        return x


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
            RandomTimeReverse(),
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

class StrongContrastiveAug:
    """
    Stronger augmentation for SimCLR-style training on log-mels.

    - Random time crop to a fixed number of frames
    - Time mask (up to ~25% of time axis)
    - Freq mask (up to ~20% of mel bins)
    - Random gain
    - Random Gaussian noise
    """
    def __init__(self, target_frames=128):
        self.target_frames = target_frames
        self.crop = RandomCrop(target_frames)
        # use a bit stronger masks than SafeContrastiveAug
        self.time_mask = RandomTimeMask(max_width=10)
        self.freq_mask = RandomFreqMask(max_width=12)
        self.gain = RandomGain(-6, 6)
        self.noise = RandomNoise(0.02)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, F, T] or [B, 1, F, T] depending on how you call it;
        # in your Dataset it's [1, F, T]
        x = self.crop(x)  # ensure fixed T = target_frames

        # time mask ~80% of the time
        if random.random() < 0.8:
            x = self.time_mask(x)

        # freq mask ~80% of the time
        if random.random() < 0.8:
            x = self.freq_mask(x)

        # gain ~50% of the time
        if random.random() < 0.5:
            x = self.gain(x)

        # noise ~70% of the time
        if random.random() < 0.7:
            x = self.noise(x)

        return x
