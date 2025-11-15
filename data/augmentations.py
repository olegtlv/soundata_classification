
import torch
import random

def get_ae_augment():
    return Compose([
        RandomTimeMask(max_width=40),
        RandomFreqMask(max_width=20),
        RandomGain(-6, 6),
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