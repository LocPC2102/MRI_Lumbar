from __future__ import annotations

import torch
from torch import nn

from src.cnn_segmentation.models.lenet5_backbone import LeNetBackbone


class LeNetUNet(nn.Module):
    """
    U-Net segmentation model with a LeNet-style backbone.

    Decoder: nearest-neighbour upsampling, concatenation with the matching
    backbone stage, then one 3x3 convolution. No batch norm, no transposed
    convolution, so every block maps directly to hardware.

    Input:
        [batch, 1, 256, 256]

    Output:
        [batch, num_classes, 256, 256]  (raw logits)
    """

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()

        self.backbone = LeNetBackbone()

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest",
        )

        # 32x32x(64 + 64) -> 32x32x32
        self.dec4 = nn.Conv2d(
            in_channels=64 + 64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 64x64x(32 + 32) -> 64x64x16
        self.dec3 = nn.Conv2d(
            in_channels=32 + 32,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 128x128x(16 + 16) -> 128x128x8
        self.dec2 = nn.Conv2d(
            in_channels=16 + 16,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 256x256x(8 + 6) -> 256x256x8
        self.dec1 = nn.Conv2d(
            in_channels=8 + 6,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 256x256x8 -> 256x256xnum_classes
        self.head = nn.Conv2d(
            in_channels=8,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.relu = nn.ReLU(inplace=False)

    def _merge(
        self,
        features: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """Upsample features by 2 and concatenate them with the skip tensor."""
        return torch.cat([self.upsample(features), skip], dim=1)

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(images)

        x = self.relu(self.dec4(self._merge(features.bottleneck, features.stage4)))
        x = self.relu(self.dec3(self._merge(x, features.stage3)))
        x = self.relu(self.dec2(self._merge(x, features.stage2)))
        x = self.relu(self.dec1(self._merge(x, features.stage1)))

        return self.head(x)