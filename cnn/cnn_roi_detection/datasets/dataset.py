import csv

import cv2
import torch
from torch.utils.data import Dataset

SITES = ("CCS", "LLS", "RLS", "LFS", "RFS")   # slot order, must match build_dataset.py
COORDS = ("xmin", "ymin", "xmax", "ymax")


class RoiDataset(Dataset):
    """
    Returns
        image  [1, S, S]  in [0, 1]
        boxes  [5, 4]     (xmin, ymin, xmax, ymax) normalized to [0, 1], row i = SITES[i]
    """

    def __init__(self, manifest_path, split: str) -> None:
        with open(manifest_path, newline="", encoding="utf-8-sig") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        image = cv2.imread(r["image"], cv2.IMREAD_GRAYSCALE)
        image = torch.from_numpy(image).float().div(255.0).unsqueeze(0)

        w, h = float(r["width"]), float(r["height"])
        boxes = torch.tensor([[float(r[f"{s}_{c}"]) for c in COORDS] for s in SITES])
        return image, boxes / torch.tensor([w, h, w, h])