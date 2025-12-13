import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    """Lightweight channel attention (SE)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)          # GAP
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ConvAutoencoderSE(nn.Module):
    """
    Parametric Conv AE with:
      - multi-scale latent (concat pooled features from multiple encoder stages)
      - channel attention (SE) per stage
    Keeps same interface: forward() -> (recon, z), encode_only() -> z

    NOTE:
      input_hw should match your actual input H,W. For your case: (64,126).
    """

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
        output_activation: str = "sigmoid",  # "sigmoid" or "none"
        input_hw=(64, 126),                  # (H, W) used to build FC layers
        attn_reduction: int = 16,            # SE reduction ratio
        ms_stages=(1, 2),                    # which encoder stages to use for multiscale (0-based)
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

        # --- build encoder ---
        enc_layers = []
        bn_layers = []
        attn_layers = []
        prev_c = in_channels
        for c in self.channels:
            enc_layers.append(nn.Conv2d(prev_c, c, self.ks, stride=self.stride, padding=self.padding))
            bn_layers.append(nn.BatchNorm2d(c) if use_bn else nn.Identity())
            attn_layers.append(SqueezeExcite(c, reduction=attn_reduction))
            prev_c = c
        self.enc_convs = nn.ModuleList(enc_layers)
        self.enc_bns = nn.ModuleList(bn_layers)
        self.enc_attn = nn.ModuleList(attn_layers)

        # --- compute stage shapes for input_hw (for decoder + optional sanity) ---
        h, w = self.input_hw
        stage_hws = []
        for _ in self.channels:
            h = self._conv_out_dim(h, self.ks, self.stride, self.padding)
            w = self._conv_out_dim(w, self.ks, self.stride, self.padding)
            stage_hws.append((h, w))
        self._stage_hws = stage_hws

        # bottleneck (last stage) sizes for decoder reshape
        self._bottleneck_hw = self._stage_hws[-1]
        self._bottleneck_c = self.channels[-1]
        self._flat_dim = self._bottleneck_c * self._bottleneck_hw[0] * self._bottleneck_hw[1]

        # --- multi-scale latent: GAP each selected stage, concat, then project to latent_dim ---
        ms_dim = 0
        for i, c in enumerate(self.channels):
            if i in self.ms_stages:
                ms_dim += c
        if ms_dim == 0:
            raise ValueError(f"ms_stages={self.ms_stages} selects no stages. Use indices in [0..{len(self.channels)-1}].")

        self.ms_proj = nn.Sequential(
            nn.Linear(ms_dim, latent_dim),
        )

        # --- build decoder (same as before) ---
        self.fc_dec = nn.Linear(latent_dim, self._flat_dim)

        dec_layers = []
        dec_bn_layers = []
        rev_channels = list(self.channels)[::-1]
        for i in range(len(rev_channels) - 1):
            in_c = rev_channels[i]
            out_c = rev_channels[i + 1]
            dec_layers.append(nn.ConvTranspose2d(in_c, out_c, self.ks, stride=self.stride, padding=self.padding))
            dec_bn_layers.append(nn.BatchNorm2d(out_c) if use_bn else nn.Identity())

        self.dec_convs = nn.ModuleList(dec_layers)
        self.dec_bns = nn.ModuleList(dec_bn_layers)
        self.dec_final = nn.ConvTranspose2d(
            rev_channels[-1], in_channels, self.ks, stride=self.stride, padding=self.padding
        )

    @staticmethod
    def _conv_out_dim(in_dim: int, k: int, s: int, p: int, d: int = 1) -> int:
        return (in_dim + 2 * p - d * (k - 1) - 1) // s + 1

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # collect multi-scale pooled vectors from selected stages
        pooled = []
        for i, (conv, bn, attn) in enumerate(zip(self.enc_convs, self.enc_bns, self.enc_attn)):
            x = conv(x)
            x = bn(x)
            x = F.relu(x, inplace=True)
            x = attn(x)

            if i in self.ms_stages:
                # GAP -> [B, C]
                pooled.append(x.mean(dim=(2, 3)))

        ms = torch.cat(pooled, dim=1)  # [B, sum(C_i)]
        z = self.ms_proj(ms)           # [B, latent_dim]
        return z

    def decode(self, z: torch.Tensor, target_shape=None) -> torch.Tensor:
        x = F.relu(self.fc_dec(z), inplace=True)
        h, w = self._bottleneck_hw
        x = x.view(-1, self._bottleneck_c, h, w)

        for deconv, bn in zip(self.dec_convs, self.dec_bns):
            x = deconv(x)
            x = bn(x)
            x = F.relu(x, inplace=True)

        x = self.dec_final(x)
        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)

        if target_shape is not None:
            x = F.interpolate(x, size=target_shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        if self.normalize_latent:
            z = F.normalize(z, dim=1)
        recon = self.decode(z, target_shape=x.shape)
        return recon, z

    def encode_only(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return F.normalize(z, dim=1) if self.normalize_latent else z
