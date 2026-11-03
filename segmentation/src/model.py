"""A small U-Net for sea/sky/obstacle segmentation.

Deliberately small (base width 16, 3 down/up stages) so it trains fast on
CPU on ~1300 images — this is a portfolio baseline, not a production
model. Input: (B, 3, H, W) in [0, 1]. Output: (B, num_classes, H, W)
logits.
"""
import torch
from torch import nn


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    def __init__(self, num_classes=3, base_width=16):
        super().__init__()
        w = base_width

        self.enc1 = _conv_block(3, w)
        self.enc2 = _conv_block(w, w * 2)
        self.enc3 = _conv_block(w * 2, w * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _conv_block(w * 4, w * 8)

        self.up3 = nn.ConvTranspose2d(w * 8, w * 4, 2, stride=2)
        self.dec3 = _conv_block(w * 8, w * 4)
        self.up2 = nn.ConvTranspose2d(w * 4, w * 2, 2, stride=2)
        self.dec2 = _conv_block(w * 4, w * 2)
        self.up1 = nn.ConvTranspose2d(w * 2, w, 2, stride=2)
        self.dec1 = _conv_block(w * 2, w)

        self.head = nn.Conv2d(w, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)
