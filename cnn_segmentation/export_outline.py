"""Draw a contour line around each predicted object (PE, CC, IVD) on the MRI slice.

Run from the project root:
    python -m src.cnn_segmentation.export_outline --input tesst
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.cnn_segmentation.datasets.transforms import CLASS_NAMES, NUM_CLASSES
from src.cnn_segmentation.models.lenet5_unet import LeNetUNet
from src.cnn_segmentation.predict import IMAGE_SUFFIXES, load_sample, predict

LINE_COLORS = ["", "red", "cyan", "yellow"]   # PE, CC, IVD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="src/cnn_segmentation/models/checkpoints_4cls/lenet_unet_best.pt")
    ap.add_argument("--input", default="tesst")
    ap.add_argument("--out-dir", default="src/cnn_segmentation/outlines")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--width", type=float, default=1.5, help="line width")
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
        labels = predict(model, x, device)

        image = Image.open(path).convert("L")
        w, h = image.size
        labels = np.array(Image.fromarray(labels.astype(np.uint8)).resize((w, h), Image.NEAREST))

        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(image, cmap="gray")
        for c in range(1, NUM_CLASSES):
            if (labels == c).any():
                ax.contour((labels == c).astype(float), levels=[0.5],
                           colors=LINE_COLORS[c], linewidths=args.width)
        ax.axis("off")
        fig.savefig(out_dir / f"{path.stem}_outline.png", dpi=100)
        plt.close(fig)
        print(f"saved {path.stem}")


if __name__ == "__main__":
    main()