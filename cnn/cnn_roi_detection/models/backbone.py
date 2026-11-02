import torch
from torch import nn


class LeNetBackbone(nn.Module):
    """256x256x1 -> 16x16x32 (stride 16)."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.out_channels = 32

        # 256x256xC -> 256x256x8 -> 128x128x8
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 128x128x8 -> 128x128x16 -> 64x64x16
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 64x64x16 -> 64x64x32 -> 32x32x32
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 32x32x32 -> 32x32x32 -> 16x16x32
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 16x16x32 -> 16x16x32 (two more convs for a larger receptive field)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu(self.conv1(images)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        return x