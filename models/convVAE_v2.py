import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Simple residual block:
    conv -> BN -> ReLU -> conv -> BN + skip -> ReLU
    Used in the encoder. Does NOT bypass the bottleneck.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1,
                                  stride=stride, bias=False)
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


class ConvVAE_v2(nn.Module):
    """
    Conv VAE v2 for log-mel spectrograms.

    Encoder:
      - 4 residual downsampling stages: 1 -> 32 -> 64 -> 128 -> 256
      - flatten -> (mu, logvar) in R^latent_dim

    Decoder (weaker than encoder):
      - z -> linear -> feature map [B, C_enc, H_enc, W_enc]
      - 4 ConvTranspose2d upsampling stages with *no BatchNorm*,
        and modest channels: 256 -> 128 -> 64 -> 32 -> 1

    No U-Net style skips, so the bottleneck stays informative.

    Interface:
      - forward(x) -> (recon, mu, logvar, z)
      - encode(x)  -> (mu, logvar, z)

    input_size should match your logmel shape, e.g. (1, 64, 126)
    for UrbanSound8k with your precomputed features.
    """
    def __init__(self, latent_dim=128, input_size=(1, 64, 126)):
        super().__init__()
        in_ch = input_size[0]

        # -----------------------
        # Encoder: residual downsampling
        # -----------------------
        self.enc1 = ResBlock(in_ch,   32, stride=2)  # [B,32, F/2,   T/2]
        self.enc2 = ResBlock(32,      64, stride=2)  # [B,64, F/4,   T/4]
        self.enc3 = ResBlock(64,     128, stride=2)  # [B,128,F/8,   T/8]
        self.enc4 = ResBlock(128,    256, stride=2)  # [B,256,F/16,  T/16]

        # figure out encoder output shape dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)
            h = self._encode_conv(dummy)
            self.enc_out_shape = h.shape[1:]     # (C_enc, H_enc, W_enc)
            flat_dim = h.numel()

        # latent: mean + logvar
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # decoder input from latent
        self.fc_dec = nn.Linear(latent_dim, flat_dim)

        # -----------------------
        # Decoder: weaker than encoder
        #   (no BN, smaller channels)
        # -----------------------
        C_enc, H_enc, W_enc = self.enc_out_shape

        # You can reduce channels here to weaken decoder,
        # but keeping 256 at start is fine – weak comes from no BN & fewer ops.
        self.dec_conv1 = nn.ConvTranspose2d(
            C_enc, 128, kernel_size=4, stride=2, padding=1
        )  # [B,128, 2H,  2W]
        self.dec_conv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # [B,64,  4H,  4W]
        self.dec_conv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # [B,32,  8H,  8W]
        self.dec_conv4 = nn.ConvTranspose2d(
            32, 1, kernel_size=4, stride=2, padding=1
        )  # [B,1,  16H, 16W]  (we'll crop in the loss if needed)

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
        returns: (mu, logvar, z)
        """
        h = self._encode_conv(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def reparameterize(self, mu, logvar):
        """
        z = mu + eps * sigma, eps ~ N(0, I)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # at eval time you usually use mean
            return mu

    # -----------------------
    # Decoder
    # -----------------------
    def decode(self, z):
        """
        z: [B, latent_dim]
        returns: recon [B,1,F',T'] (we don't guarantee exact size match;
                 your loss already crops to match x_clean)
        """
        B = z.size(0)
        flat = self.fc_dec(z)
        h = flat.view(B, *self.enc_out_shape)  # [B,C_enc,H_enc,W_enc]

        # no BN: weaker decoder, encourages latent to carry more info
        h = F.relu(self.dec_conv1(h), inplace=True)
        h = F.relu(self.dec_conv2(h), inplace=True)
        h = F.relu(self.dec_conv3(h), inplace=True)
        h = self.dec_conv4(h)  # [B,1,H_out,W_out]

        # if your specs are in [0,1], you can clamp or sigmoid:
        # h = torch.sigmoid(h)
        return h

    # -----------------------
    # Forward
    # -----------------------
    def forward(self, x):
        """
        x: [B,1,F,T]
        returns: (recon, mu, logvar, z)
        """
        h = self._encode_conv(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar#, z
