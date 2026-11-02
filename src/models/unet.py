"""Segmentation heads trained on frozen inputs.

- PixelMLP (baselines B1/B4): two 1x1 convolutions, so each pixel is classified
  from its own features only (no spatial context).
- LightUNet (baselines B2/B5/B6 and the fusion variants): a 1x1 convolution maps
  the C input channels to the first width, then a 3-level UNet (two 3x3
  conv-BN-ReLU per level, max-pooling down, bilinear upsampling with skip
  connections), Dropout2d(0.1) at the bottom level and a 1x1 convolution to the
  classes. With 128 input channels and 19 classes, widths (32, 64, 128) give
  485,267 parameters and (64, 128, 256) give 1,928,979 (checked in tests).

Input (B, C, H, W) with H and W divisible by 4 -> logits (B, n_classes, H, W).
"""

import torch
import torch.nn.functional as F
from torch import nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class LightUNet(nn.Module):
    def __init__(self, in_channels, n_classes=19, widths=(32, 64, 128), dropout=0.1):
        super().__init__()
        w1, w2, w3 = widths
        self.inp = nn.Conv2d(in_channels, w1, 1)
        self.enc1 = conv_block(w1, w1)
        self.enc2 = conv_block(w1, w2)
        self.mid = conv_block(w2, w3)
        self.drop = nn.Dropout2d(dropout)
        self.dec2 = conv_block(w3 + w2, w2)
        self.dec1 = conv_block(w2 + w1, w1)
        self.out = nn.Conv2d(w1, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(self.inp(x))
        e2 = self.enc2(F.max_pool2d(e1, 2))
        m = self.drop(self.mid(F.max_pool2d(e2, 2)))
        d2 = self.dec2(torch.cat([up(m, e2), e2], 1))
        d1 = self.dec1(torch.cat([up(d2, e1), e1], 1))
        return self.out(d1)


def up(x, like):
    return F.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=False)


class PixelMLP(nn.Module):
    def __init__(self, in_channels, n_classes=19, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, hidden, 1), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                 nn.Conv2d(hidden, n_classes, 1))

    def forward(self, x):
        return self.net(x)


def build_head(name, in_channels, n_classes=19, widths=(32, 64, 128), hidden=128, dropout=0.1):
    if name == "unet":
        return LightUNet(in_channels, n_classes, tuple(widths), dropout)
    if name == "mlp":
        return PixelMLP(in_channels, n_classes, hidden, dropout)
    raise ValueError(f"unknown head {name!r}: use 'unet' or 'mlp'")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
