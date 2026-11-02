"""Export predicted masks (Background, PE, CC, IVD) as PNG files.

Run from the project root:
    python -m src.cnn_segmentation.export_masks --input tesst
    python -m src.cnn_segmentation.export_masks --input tesst --per-class
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.cnn_segmentation.datasets.transforms import CLASS_NAMES, NUM_CLASSES, RAW_VALUES
from src.cnn_segmentation.models.lenet5_unet import LeNetUNet
from src.cnn_segmentation.predict import IMAGE_SUFFIXES, load_sample, predict

RAW = np.array(RAW_VALUES, dtype=np.uint8)   # class id -> raw pixel value (0, 50, 100, 200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="src/cnn_segmentation/models/checkpoints/lenet_unet_best.pt")
    ap.add_argument("--input", default="tesst", help="image file or folder")
    ap.add_argument("--out-dir", default="src/cnn_segmentation/masks")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--per-class", action="store_true", help="also save one binary mask per class")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LeNetUNet(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    source = Path(args.input)
    paths = [source] if source.is_file() else sorted(
        p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        _, x, _ = load_sample(path, None, args.size)
        labels = predict(model, x, device)                      # [size, size], values 0..3
        original_size = Image.open(path).size                   # (w, h)

        def save(array, name):
            Image.fromarray(array).resize(original_size, Image.NEAREST).save(out_dir / name)

        save(RAW[labels], f"{path.stem}_mask.png")              # same format as the training masks
        if args.per_class:
            for c in range(1, NUM_CLASSES):
                save(((labels == c) * 255).astype(np.uint8), f"{path.stem}_{CLASS_NAMES[c]}.png")
        print(f"saved {path.stem}")


if __name__ == "__main__":
    main()