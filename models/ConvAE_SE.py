import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGate(nn.Module):
    """Simple spatial attention: produces a (B,1,H,W) mask."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.sigmoid(self.conv(x))
        return x * m


class ResidualSE(nn.Module):
    """SE with residual path to prevent over-suppression."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x + x * s  # residual attention


class ConvAutoencoderSE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels=(32, 64, 96),
        latent_dim: int = 128,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        use_bn: bool = True,
        normalize_latent: bool = True,
        output_activation: str = "sigmoid",
        input_hw=(64, 126),

        # new knobs (safe defaults)
        ms_stages=(0, 1, 2),          # use all stages by default
        ms_pool: str = "avgmax",      # "avg" or "avgmax"
        attn_type: str = "spatial",   # "spatial" or "se" or "none"
        attn_reduction: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = tuple(channels)
        self.latent_dim = latent_dim
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bn = use_bn
        self.normalize_latent = normalize_latent
        self.output_activation = output_activation.lower()
        self.input_hw = tuple(input_hw)

        self.ms_stages = tuple(ms_stages)
        self.ms_pool = ms_pool.lower()
        self.attn_type = attn_type.lower()

        # --- encoder ---
        self.enc_convs = nn.ModuleList()
        self.enc_norms = nn.ModuleList()
        self.enc_attn = nn.ModuleList()

        prev_c = in_channels
        for c in self.channels:
            self.enc_convs.append(nn.Conv2d(prev_c, c, self.ks, stride=self.stride, padding=self.padding))
            self.enc_norms.append(nn.BatchNorm2d(c) if use_bn else nn.Identity())

            if self.attn_type == "spatial":
                self.enc_attn.append(SpatialGate(c))
            elif self.attn_type == "se":
                self.enc_attn.append(ResidualSE(c, reduction=attn_reduction))
            else:
                self.enc_attn.append(nn.Identity())

            prev_c = c

        # bottleneck shape for decoder
        h, w = self.input_hw
        for _ in self.channels:
            h = self._conv_out_dim(h, self.ks, self.stride, self.padding)
            w = self._conv_out_dim(w, self.ks, self.stride, self.padding)
        self._bottleneck_hw = (h, w)
        self._bottleneck_c = self.channels[-1]
        self._flat_dim = self._bottleneck_c * h * w

        # multiscale projection
        ms_dim = 0
        for i, c in enumerate(self.channels):
            if i in self.ms_stages:
                ms_dim += c * (2 if self.ms_pool == "avgmax" else 1)
        if ms_dim == 0:
            raise ValueError("ms_stages selects no stages.")

        self.ms_proj = nn.Linear(ms_dim, latent_dim)

        # --- decoder (same as before) ---
        self.fc_dec = nn.Linear(latent_dim, self._flat_dim)

        rev_channels = list(self.channels)[::-1]
        self.dec_convs = nn.ModuleList()
        self.dec_norms = nn.ModuleList()
        for i in range(len(rev_channels) - 1):
            self.dec_convs.append(nn.ConvTranspose2d(rev_channels[i], rev_channels[i + 1],
                                                    self.ks, stride=self.stride, padding=self.padding))
            self.dec_norms.append(nn.BatchNorm2d(rev_channels[i + 1]) if use_bn else nn.Identity())

        self.dec_final = nn.ConvTranspose2d(rev_channels[-1], in_channels,
                                            self.ks, stride=self.stride, padding=self.padding)

    @staticmethod
    def _conv_out_dim(in_dim: int, k: int, s: int, p: int, d: int = 1) -> int:
        return (in_dim + 2 * p - d * (k - 1) - 1) // s + 1

    def _pool_stage(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> [B,C] or [B,2C]
        avg = x.mean(dim=(2, 3))
        if self.ms_pool == "avgmax":
            mx = x.amax(dim=(2, 3))
            return torch.cat([avg, mx], dim=1)
        return avg

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pooled = []
        for i, (conv, norm, attn) in enumerate(zip(self.enc_convs, self.enc_norms, self.enc_attn)):
            x = F.relu(norm(conv(x)), inplace=True)
            x = attn(x)
            if i in self.ms_stages:
                pooled.append(self._pool_stage(x))
        ms = torch.cat(pooled, dim=1)
        z = self.ms_proj(ms)
        return z

    def decode(self, z: torch.Tensor, target_shape=None) -> torch.Tensor:
        x = F.relu(self.fc_dec(z), inplace=True)
        h, w = self._bottleneck_hw
        x = x.view(-1, self._bottleneck_c, h, w)

        for deconv, norm in zip(self.dec_convs, self.dec_norms):
            x = F.relu(norm(deconv(x)), inplace=True)

        x = self.dec_final(x)
        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        if target_shape is not None:
            x = F.interpolate(x, size=target_shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor):
        z_raw = self.encode(x)
        # decode from raw to avoid unit-norm constraint harming recon/gradients
        recon = self.decode(z_raw, target_shape=x.shape)
        z = F.normalize(z_raw, dim=1) if self.normalize_latent else z_raw
        return recon, z

    def encode_only(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return F.normalize(z, dim=1) if self.normalize_latent else z
