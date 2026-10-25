"""3-D U-Net, four levels."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(i, o, 3, padding=1, bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True),
            nn.Conv3d(o, o, 3, padding=1, bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool3d(2), DoubleConv(i, o))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Transposed-conv upsampling, padded to the skip's odd sizes, concatenated, convolved."""

    def __init__(self, i, o):
        super().__init__()
        self.up = nn.ConvTranspose3d(i, i // 2, 2, stride=2)
        self.conv = DoubleConv(i, o)

    def forward(self, x, skip):
        x = self.up(x)
        dz, dy, dx = (s - t for s, t in zip(skip.shape[2:], x.shape[2:]))
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2])
        return self.conv(torch.cat([skip, x], 1))


class OutConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Conv3d(i, o, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """The paper used 64 base filters (90 M parameters, 10.6 TFLOP per frame); 16 gives the same Dice at
    5.6 M parameters and 0.7 TFLOP."""

    def __init__(self, in_ch=9, out_ch=1, base=16):
        super().__init__()
        b = base
        self.inc = DoubleConv(in_ch, b)
        self.down1, self.down2 = Down(b, 2 * b), Down(2 * b, 4 * b)
        self.down3, self.down4 = Down(4 * b, 8 * b), Down(8 * b, 16 * b)
        self.up1, self.up2 = Up(16 * b, 8 * b), Up(8 * b, 4 * b)
        self.up3, self.up4 = Up(4 * b, 2 * b), Up(2 * b, b)
        self.outc = OutConv(b, out_ch)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(self.up3(self.up2(self.up1(x5, x4), x3), x2), x1)
        return self.outc(x)


def fold_bn(model):
    """Fold each BatchNorm3d into the convolution before it (eval mode); returns a copy."""
    model = copy.deepcopy(model).eval()
    for seq in (m for m in model.modules() if isinstance(m, nn.Sequential)):
        for i in range(len(seq) - 1):
            if isinstance(seq[i], nn.Conv3d) and isinstance(seq[i + 1], nn.BatchNorm3d):
                seq[i], seq[i + 1] = fuse_conv_bn_eval(seq[i], seq[i + 1]), nn.Identity()
    return model
