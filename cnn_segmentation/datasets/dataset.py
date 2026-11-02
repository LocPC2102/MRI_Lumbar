"""Dataset that reads a split CSV (seg_train.csv / seg_val.csv / seg_test.csv)."""

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .transforms import DualToTensor


class AxialSegmentationDataset(Dataset):
    def __init__(self, csv_path, root=".", transform=None):
        self.root = Path(root)
        self.transform = transform or DualToTensor()
        with open(csv_path, newline="", encoding="utf-8") as f:
            self.rows = [r for r in csv.DictReader(f) if r["pair_valid"] == "True"]
        if not self.rows:
            raise ValueError(f"No valid samples in {csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        image = Image.open(self.root / r["image_path"]).convert("L")
        mask = Image.open(self.root / r["mask_path"]).convert("L")
        return self.transform(image, mask)