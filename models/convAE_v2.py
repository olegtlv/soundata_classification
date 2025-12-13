# models/convAE_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Simple residual block for 2D feature maps:
    conv -> BN -> ReLU -> conv -> BN + skip -> ReLU
    Used inside encoder/decoder; does NOT bypass the bottleneck.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # projection for skip if channels or stride differ
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        skip = self.skip(x)
        out = out + skip
        out = F.relu(out, inplace=True)
        return out


class ConvAutoencoder_v2(nn.Module):
    """
    Conv autoencoder v2 for log-mel spectrograms.

    - Encoder: 4 residual downsampling stages: 1→32→64→128→256
    - Bottleneck: flatten + Linear -> latent_dim
    - Decoder: Linear -> 256xH'xW' + 4 upsampling stages back to 1 channel
    - No U-Net-style encoder→decoder skips, so bottleneck stays informative.

    Interface:
      - forward(x) -> (recon, z)
      - encode(x)  -> z

    Default input_size assumes [1, 128, 128] (C,F,T).
    If your mel/time resolution differs, pass input_size=(1, F, T).
    """
    def __init__(self, latent_dim=128, input_size=(1, 128, 128)):
        super().__init__()
        in_ch = input_size[0]

        # -----------------------
        # Encoder: residual downsampling
        # -----------------------
        self.enc1 = ResBlock(in_ch,   32, stride=2)  # [B,32, F/2,   T/2]
        self.enc2 = ResBlock(32,      64, stride=2)  # [B,64, F/4,   T/4]
        self.enc3 = ResBlock(64,     128, stride=2)  # [B,128,F/8,   T/8]
        self.enc4 = ResBlock(128,    256, stride=2)  # [B,256,F/16,  T/16]

        # we need to know encoder output shape to build FC layers
        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)  # [1, C, F, T]
            h = self._encode_conv(dummy)
            self.enc_out_shape = h.shape[1:]     # (C_enc, H_enc, W_enc)
            flat_dim = h.numel()

        # bottleneck fully-connected layers
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        # -----------------------
        # Decoder: upsampling
        # -----------------------
        C_enc, H_enc, W_enc = self.enc_out_shape

        # we mirror channel sizes: 256 -> 128 -> 64 -> 32 -> 1
        self.dec_conv1 = nn.ConvTranspose2d(
            C_enc, 32, kernel_size=4, stride=2, padding=1
        )  # [B,128, 2H,   2W]
        # self.dec_bn1 = nn.BatchNorm2d(128)

        # self.dec_conv2 = nn.ConvTranspose2d(
        #     128, 64, kernel_size=4, stride=2, padding=1
        # )  # [B,64,  4H,   4W]
        # # self.dec_bn2 = nn.BatchNorm2d(64)

        self.dec_conv3 = nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1
        )  # [B,32,  8H,   8W]
        # self.dec_bn3 = nn.BatchNorm2d(32)

        self.dec_conv4 = nn.ConvTranspose2d(
            16, 1, kernel_size=4, stride=2, padding=1
        )  # [B,1,  16H,  16W] assuming 4 downsamples

        # if your input is exactly (F,T)=(128,128), and you downsample 4 times,
        # H_enc=W_enc=8 -> 16H=128, 16W=128 => perfect match.
        # If you use other input_size, adjust #stages or kernel/stride.

    # -----------------------
    # Encoder helpers
    # -----------------------
    def _encode_conv(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return x

    def encode(self, x):
        """
        x: [B,1,F,T]
        returns: z ∈ [B, latent_dim]
        """
        h = self._encode_conv(x)
        h_flat = h.view(h.size(0), -1)
        z = self.fc_mu(h_flat)
        return z

    # -----------------------
    # Decoder
    # -----------------------
    def decode(self, z):
        """
        z: [B, latent_dim]
        returns: recon [B,1,F,T]
        """
        B = z.size(0)
        flat = self.fc_dec(z)
        h = flat.view(B, *self.enc_out_shape)  # [B,C_enc,H_enc,W_enc]

        h = self.dec_conv1(h)
        # h = self.dec_bn1(h)
        h = F.relu(h, inplace=True)

        # h = self.dec_conv2(h)
        # h = self.dec_bn2(h)
        # h = F.relu(h, inplace=True)

        h = self.dec_conv3(h)
        # h = self.dec_bn3(h)
        h = F.relu(h, inplace=True)

        h = self.dec_conv4(h)
        # no activation here; your loss already handles the scale
        # if your inputs are strictly in [0,1], you *could* add sigmoid:
        # h = torch.sigmoid(h)

        return h

    # -----------------------
    # Standard forward
    # -----------------------
    def forward(self, x):
        """
        x: [B,1,F,T]
        returns: (recon, z)
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
