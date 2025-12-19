
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


class StrongContrastiveAug:
    def __init__(self, target_frames=128):
        self.crop = EnergyBiasedRandomCrop(target_frames)
        self.tshift = RandomTimeShift(max_shift=12)
        self.fshift = RandomFreqShift(max_shift=2)
        self.time_mask = RandomTimeMask(max_width=8)
        self.freq_mask = RandomFreqMask(max_width=10)
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
