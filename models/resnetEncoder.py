# models/resnet_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, trainable=False):
        super().__init__()
        if trainable:
            self.p = nn.Parameter(torch.tensor(float(p)))
        else:
            self.register_buffer("p", torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=(2, 3), keepdim=True).pow(1.0 / p)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim=128, depth="layer3", pretrained=True):
        """
        depth: which resnet stage to stop at.
               'layer2' -> fastest, lowest capacity
               'layer3' -> good tradeoff
               'layer4' -> full resnet18 (slowest)
        """
        super().__init__()

        # weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        # base = resnet18(weights=weights)
        base = resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # ----- adapt first conv to 1-channel spectrograms -----
        old_conv = base.conv1
        base.conv1 = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            # average RGB weights -> single channel
            with torch.no_grad():
                base.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        # ----- select how deep we go -----
        # resnet18 children: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        layers = [
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
        ]
        out_channels = 64  # after layer1

        if depth in ("layer2", "layer3", "layer4"):
            layers.append(base.layer2)
            out_channels = 128
        if depth in ("layer3", "layer4"):
            layers.append(base.layer3)
            out_channels = 256
        if depth == "layer4":
            layers.append(base.layer4)
            out_channels = 512

        # old_conv = base.conv1  # [out_c=64, in_c=3, ...]
        # base.conv1 = nn.Conv2d(
        #     1,
        #     old_conv.out_channels,
        #     kernel_size=old_conv.kernel_size,
        #     stride=old_conv.stride,
        #     padding=old_conv.padding,
        #     bias=False,
        # )
        # if pretrained:
        #     with torch.no_grad():
        #         base.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        self.features = nn.Sequential(*layers)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool = GeM(p=3.0, trainable=False)  # try p=3 or 4
        self.fc = nn.Linear(out_channels, latent_dim)

    def encode(self, x):
        # x: [B, 1, F, T]
        h = self.features(x)          # [B, C, H, W]
        h = self.pool(h)              # [B, C, 1, 1]
        h = torch.flatten(h, 1)       # [B, C]
        z = self.fc(h)                # [B, latent_dim]
        return z

    def forward(self, x):
        # match your ConvAE interface: return (recon, z)
        z = self.encode(x)
        return None, z

    def encode_only(self, x):
        return self.encode(x)
