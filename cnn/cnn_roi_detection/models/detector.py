import torch
from torch import nn

from .backbone import LeNetBackbone


class RoiDetector(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_sites: int = 5,
        num_classes: int = 0,
    ) -> None:
        super().__init__()

        self.num_sites = num_sites
        self.values_per_site = 5 + num_classes

        self.backbone = LeNetBackbone(
            in_channels=in_channels,
        )

        # 16x16x32 -> 16x16x(num_sites * values_per_site)
        #
        # This maps directly to a 1x1 convolution in the RTL.
        self.detection_head = nn.Conv2d(
            in_channels=self.backbone.out_channels,
            out_channels=self.num_sites * self.values_per_site,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(images)

        # Do not apply ReLU, sigmoid or softmax here.
        raw_predictions = self.detection_head(features)

        return raw_predictions


if __name__ == "__main__":
    model = RoiDetector()
    output = model(torch.zeros(2, 1, 64, 64))
    print("output shape:", tuple(output.shape))
    print("parameters  :", sum(p.numel() for p in model.parameters()))