import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    """
    Parametric Conv AE.
    Assumes input spatial dims are divisible by 2**len(channels).
    If not, decode() will still resize to match the original input via interpolate.
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
        output_activation: str = "none",  # "sigmoid" or "none"
        input_hw=(64, 128),  # (H, W) used to build FC layers
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

        # --- build encoder ---
        enc_layers = []
        bn_layers = []
        prev_c = in_channels
        for c in self.channels:
            enc_layers.append(nn.Conv2d(prev_c, c, self.ks, stride=self.stride, padding=self.padding))
            bn_layers.append(nn.BatchNorm2d(c) if use_bn else nn.Identity())
            prev_c = c
        self.enc_convs = nn.ModuleList(enc_layers)
        self.enc_bns = nn.ModuleList(bn_layers)

        # compute bottleneck feature size for FC based on input_hw and conv params
        h, w = self.input_hw
        for _ in self.channels:
            h = self._conv_out_dim(h, self.ks, self.stride, self.padding)
            w = self._conv_out_dim(w, self.ks, self.stride, self.padding)
        self._bottleneck_hw = (h, w)
        self._bottleneck_c = self.channels[-1]
        self._flat_dim = self._bottleneck_c * h * w

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(self._flat_dim, latent_dim)

        # --- build decoder ---
        self.fc_dec = nn.Linear(latent_dim, self._flat_dim)

        dec_layers = []
        dec_bn_layers = []
        rev_channels = list(self.channels)[::-1]  # e.g. 64,32,16
        for i in range(len(rev_channels) - 1):
            in_c = rev_channels[i]
            out_c = rev_channels[i + 1]
            dec_layers.append(
                nn.ConvTranspose2d(in_c, out_c, self.ks, stride=self.stride, padding=self.padding)
            )
            dec_bn_layers.append(nn.BatchNorm2d(out_c) if use_bn else nn.Identity())

        # final layer to reconstruct in_channels
        self.dec_convs = nn.ModuleList(dec_layers)
        self.dec_bns = nn.ModuleList(dec_bn_layers)
        self.dec_final = nn.ConvTranspose2d(rev_channels[-1], in_channels, self.ks, stride=self.stride, padding=self.padding)

    @staticmethod
    def _conv_out_dim(in_dim: int, k: int, s: int, p: int, d: int = 1) -> int:
        # PyTorch Conv2d output: floor((in + 2p - d*(k-1) - 1)/s + 1)
        return (in_dim + 2 * p - d * (k - 1) - 1) // s + 1

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for conv, bn in zip(self.enc_convs, self.enc_bns):
            x = F.relu(bn(conv(x)))
        x = self.flatten(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z: torch.Tensor, target_shape=None) -> torch.Tensor:
        x = F.relu(self.fc_dec(z))
        h, w = self._bottleneck_hw
        x = x.view(-1, self._bottleneck_c, h, w)

        for deconv, bn in zip(self.dec_convs, self.dec_bns):
            x = F.relu(bn(deconv(x)))

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
        # return F.normalize(z, dim=1)
