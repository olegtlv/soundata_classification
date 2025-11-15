# models/convVAE_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    """
    Robust Conv VAE using upsample+conv decoder (avoids ConvTranspose2d shape issues).
    Dynamically initializes latent layers after first forward.
    """
    def __init__(self, latent_dim=128, hidden_channels=[32,64,128,256]):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        # Encoder convs
        chs = hidden_channels
        self.enc1 = nn.Conv2d(1, chs[0], kernel_size=3, stride=2, padding=1)  # /2
        self.bn1 = nn.BatchNorm2d(chs[0])
        self.enc2 = nn.Conv2d(chs[0], chs[1], kernel_size=3, stride=2, padding=1)  # /4
        self.bn2 = nn.BatchNorm2d(chs[1])
        self.enc3 = nn.Conv2d(chs[1], chs[2], kernel_size=3, stride=2, padding=1)  # /8
        self.bn3 = nn.BatchNorm2d(chs[2])
        self.enc4 = nn.Conv2d(chs[2], chs[3], kernel_size=3, stride=2, padding=1)  # /16
        self.bn4 = nn.BatchNorm2d(chs[3])

        self.flatten = nn.Flatten()

        # These will be created on first forward and moved to device
        self._feat_dim = None
        self._enc_C = None
        self._enc_H = None
        self._enc_W = None

        self.fc_mu = None
        self.fc_logvar = None
        self.fc_dec = None

        # Decoder convs (will be created in _init_decoder as regular nn.Conv2d)
        self.dec_conv1 = None
        self.dec_conv2 = None
        self.dec_conv3 = None
        self.dec_conv4 = None
        self.final_conv = None

    def _device(self):
        # helper to get model device
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def _init_latent_and_decoder(self, feat_tensor):
        # feat_tensor: tensor after last encoder conv, shape [B, C, H, W]
        C, H, W = feat_tensor.shape[1:]
        feat_dim = C * H * W
        self._enc_C, self._enc_H, self._enc_W = C, H, W
        self._feat_dim = feat_dim

        dev = self._device()

        # latent mapping
        self.fc_mu = nn.Linear(feat_dim, self.latent_dim).to(dev)
        self.fc_logvar = nn.Linear(feat_dim, self.latent_dim).to(dev)
        self.fc_dec = nn.Linear(self.latent_dim, feat_dim).to(dev)

        # decoder convs: we'll use convs with upsample to double spatial dims each step
        # define convs that map channels -> channels/2 ... -> 1
        # Mirror encoder channel layout if possible
        out_chs = list(reversed(self.hidden_channels))
        # conv after upsample: in_ch -> out_ch
        self.dec_conv1 = nn.Conv2d(out_chs[0], out_chs[1], kernel_size=3, padding=1).to(dev)
        self.dec_conv2 = nn.Conv2d(out_chs[1], out_chs[2], kernel_size=3, padding=1).to(dev)
        self.dec_conv3 = nn.Conv2d(out_chs[2], out_chs[3], kernel_size=3, padding=1).to(dev)
        # final conv to 1 channel
        self.final_conv = nn.Conv2d(out_chs[3], 1, kernel_size=3, padding=1).to(dev)

        # BatchNorm layers for decoder (optional)
        self.d_bn1 = nn.BatchNorm2d(out_chs[1]).to(dev)
        self.d_bn2 = nn.BatchNorm2d(out_chs[2]).to(dev)
        self.d_bn3 = nn.BatchNorm2d(out_chs[3]).to(dev)

    def encode(self, x):
        x = F.relu(self.bn1(self.enc1(x)))
        x = F.relu(self.bn2(self.enc2(x)))
        x = F.relu(self.bn3(self.enc3(x)))
        x = F.relu(self.bn4(self.enc4(x)))

        # initialize latent & decoder using this sample's shape (only once)
        if self._feat_dim is None:
            self._init_latent_and_decoder(x)

        flat = self.flatten(x)  # [B, feat_dim]
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z: [B, latent_dim]
        B = z.size(0)
        x = self.fc_dec(z)  # [B, feat_dim]
        x = x.view(B, self._enc_C, self._enc_H, self._enc_W)  # reshape to encoder feature map shape

        # decoder - upsample + conv blocks
        # We'll perform 3 upsample+conv steps (matches encoder 4 convs -> 4 upsample steps).
        # Note: number of steps must be consistent with encoder depth; designed to match hidden_channels length.
        # 1) upsample (H,W) -> (H*2, W*2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.d_bn1(self.dec_conv1(x)))

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.d_bn2(self.dec_conv2(x)))

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.d_bn3(self.dec_conv3(x)))

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.final_conv(x)  # raw output (may be larger/smaller by a couple frames)

        # optionally apply sigmoid if your inputs are normalized to [0,1]
        # return logits to let trainer decide (but we'll sigmoid here for typical mel dB normalization in [0,1])
        out = torch.sigmoid(x)
        return out

    def match_size(self, recon, x):
        """
        Crop or pad recon so it matches x shape: [B, C, H, W]
        """
        _, _, Ht, Wt = x.shape
        _, _, Hr, Wr = recon.shape

        # crop if bigger
        if Hr > Ht:
            recon = recon[:, :, :Ht, :]
        elif Hr < Ht:
            pad_h = Ht - Hr
            recon = F.pad(recon, (0,0,0,pad_h))

        if Wr > Wt:
            recon = recon[:, :, :, :Wt]
        elif Wr < Wt:
            pad_w = Wt - Wr
            recon = F.pad(recon, (0,pad_w,0,0))

        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon = self.match_size(recon, x)   # robustify final shape
        return recon, mu, logvar
