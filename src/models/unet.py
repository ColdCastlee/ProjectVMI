"""Segmentation heads trained on frozen inputs.

- LightUNet (baselines B2/B5/B6 and the fusion variants): a 1x1 convolution maps
  the C input channels to the first width, then a 3-level UNet (two 3x3
  conv-BN-ReLU per level, max-pooling down, bilinear upsampling with skip
  connections), Dropout2d(0.1) at the bottom level and a 1x1 convolution to the
  classes. With 128 input channels and 19 classes, widths (32, 64, 128) give
  485,267 parameters and (64, 128, 256) give 1,928,979 (checked in tests).
- GatedFusionUNet (Method 1, M1-4): two sources (e.g. TESSERA and AlphaEarth) are
  projected by 1x1 convolutions to the first width, mixed pixel by pixel as
  z = a * f1(x1) + (1 - a) * f2(x2) with a = sigmoid(g([x1; x2])), one weight a
  per pixel, and z goes through the LightUNet.

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


class GatedFusionUNet(nn.Module):
    """Pixel-wise gate between two sources stacked on the channels, then a LightUNet."""

    def __init__(self, splits, n_classes=19, widths=(32, 64, 128), dropout=0.1):
        super().__init__()
        if len(splits) != 2:
            raise ValueError(f"the gated head mixes two sources, got channels {list(splits)}")
        self.splits = [int(c) for c in splits]
        self.proj = nn.ModuleList(nn.Conv2d(c, widths[0], 1) for c in self.splits)
        self.gate = nn.Conv2d(sum(self.splits), 1, 1)
        self.unet = LightUNet(widths[0], n_classes, tuple(widths), dropout)

    def alpha(self, x):
        """(B, 1, H, W) weight of the first source at every pixel."""
        return torch.sigmoid(self.gate(x))

    def forward(self, x):
        x1, x2 = torch.split(x, self.splits, dim=1)
        a = self.alpha(x)
        return self.unet(a * self.proj[0](x1) + (1 - a) * self.proj[1](x2))


def build_head(name, in_channels, n_classes=19, widths=(32, 64, 128), dropout=0.1, splits=None):
    """splits: input channels of each source (needed by the gated head only)."""
    if name == "unet":
        return LightUNet(in_channels, n_classes, tuple(widths), dropout)
    if name == "gated":
        if splits is None or sum(splits) != in_channels:
            raise ValueError(f"gated head: channels per source {splits} do not add up to {in_channels}")
        return GatedFusionUNet(splits, n_classes, tuple(widths), dropout)
    raise ValueError(f"unknown head {name!r}: use 'unet' or 'gated'")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
