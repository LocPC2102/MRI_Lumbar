from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class BackboneFeatures(NamedTuple):
    """Feature maps of the backbone, from high to low resolution."""

    stage1: torch.Tensor      # [batch,  6, 256, 256]
    stage2: torch.Tensor      # [batch, 16, 128, 128]
    stage3: torch.Tensor      # [batch, 32,  64,  64]
    stage4: torch.Tensor      # [batch, 64,  32,  32]
    bottleneck: torch.Tensor  # [batch, 64,  16,  16]


class LeNetBackbone(nn.Module):
    """
    LeNet-style encoder for 256x256 grayscale MRI slices.

    Same building blocks as LeNet-5 (convolution, ReLU, max pooling),
    stacked deeper so the receptive field can cover a whole vertebra.
    The stage outputs are returned so a U-Net decoder can use them as skips.

    Input:
        [batch, 1, 256, 256]

    Output:
        BackboneFeatures (see above), bottleneck is [batch, 64, 16, 16]
    """

    def __init__(self) -> None:
        super().__init__()

        # 256x256x1 -> 256x256x6
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )

        # 256x256x6 -> 128x128x6
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # 128x128x6 -> 128x128x16
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )

        # 128x128x16 -> 64x64x16
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # 64x64x16 -> 64x64x32
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 64x64x32 -> 32x32x32
        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # 32x32x32 -> 32x32x64
        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 32x32x64 -> 16x16x64
        self.pool4 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # 16x16x64 -> 16x16x64
        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.relu = nn.ReLU(inplace=False)

    def forward(
        self,
        images: torch.Tensor,
    ) -> BackboneFeatures:
        stage1 = self.relu(self.conv1(images))
        stage2 = self.relu(self.conv2(self.pool1(stage1)))
        stage3 = self.relu(self.conv3(self.pool2(stage2)))
        stage4 = self.relu(self.conv4(self.pool3(stage3)))
        bottleneck = self.relu(self.conv5(self.pool4(stage4)))

        return BackboneFeatures(
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            stage4=stage4,
            bottleneck=bottleneck,
        )