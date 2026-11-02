"""Predict with a trained LeNetUNet and draw the result on the MRI slice.

Run from the project root:
    python -m src.cnn_segmentation.predict                      # every image in tesst/
    python -m src.cnn_segmentation.predict --input path/to/folder_or_image
    python -m src.cnn_segmentation.predict --split test --num 6 # dataset samples + ground truth

Each result is saved as a PNG: MRI | prediction (| ground truth for --split).
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from PIL import Image

from src.cnn_segmentation.datasets.dataset import AxialSegmentationDataset
from src.cnn_segmentation.datasets.transforms import (
    CLASS_NAMES, NUM_CLASSES, DualResize, DualToTensor)
from src.cnn_segmentation.models.lenet5_unet import LeNetUNet

# RGB colour per class id (Background is never drawn)
CLASS_COLORS = [(0, 0, 0), (230, 57, 70), (42, 157, 143), (66, 135, 245)]
ALPHA = 0.45


def load_sample(image_path, mask_path, size):
    """Return the resized slice (uint8), the model input and the label map (or None)."""
    image = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L") if mask_path else Image.new("L", image.size, 0)

    image, mask = DualResize(size)(image, mask)
    x, y = DualToTensor()(image, mask)
    return np.array(image), x, (y.numpy() if mask_path else None)


@torch.no_grad()
def predict(model, x, device):
    return model(x.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()


def colorize(image, labels):
    """Blend class colours onto the grayscale slice. Background stays untouched."""
    rgb = np.stack([image] * 3, axis=-1).astype(np.float32)
    for class_id in range(1, NUM_CLASSES):
        region = labels == class_id
        rgb[region] = (1 - ALPHA) * rgb[region] + ALPHA * np.array(CLASS_COLORS[class_id])
    return rgb.astype(np.uint8)


def dice_per_class(prediction, target):
    """Dice of each foreground class for one slice (NaN if the class is absent in both)."""
    scores = []
    for class_id in range(1, NUM_CLASSES):
        pred, true = prediction == class_id, target == class_id
        total = pred.sum() + true.sum()
        scores.append(2 * (pred & true).sum() / total if total else float("nan"))
    return scores


def save_figure(image, prediction, target, title, out_path, show):
    panels = [("MRI", np.stack([image] * 3, axis=-1))]

    if target is not None:
        panels.append(("Ground truth", colorize(image, target)))
        scores = dice_per_class(prediction, target)
        dice_text = "  ".join(f"{name} {score:.2f}" for name, score in zip(CLASS_NAMES[1:], scores))
        panels.append((f"Prediction\nDice: {dice_text}", colorize(image, prediction)))
    else:
        panels.append(("Prediction", colorize(image, prediction)))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.4))
    for ax, (name, panel) in zip(np.atleast_1d(axes), panels):
        ax.imshow(panel)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    legend = [Patch(color=np.array(CLASS_COLORS[c]) / 255, label=CLASS_NAMES[c])
              for c in range(1, NUM_CLASSES)]
    fig.legend(handles=legend, loc="lower center", ncol=len(legend))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def pick_samples(args):
    """Return a list of (image_path, mask_path or None, name)."""
    if args.split:
        dataset = AxialSegmentationDataset(f"DATA/processed/seg_{args.split}.csv")
        rows = random.Random(args.seed).sample(dataset.rows, min(args.num, len(dataset.rows)))
        return [(Path(r["image_path"]), Path(r["mask_path"]),
                 f"{r['patient_id']}_{Path(r['image_path']).stem}") for r in rows]

    source = Path(args.input)
    paths = [source] if source.is_file() else sorted(
        p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No images found in {source}")
    return [(path, None, path.stem) for path in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="src/cnn_segmentation/models/checkpoints/lenet_unet_best.pt")
    ap.add_argument("--input", default="tesst", help="image file or folder of images (no ground truth)")
    ap.add_argument("--split", choices=["train", "val", "test"], default=None,
                    help="use random samples of a split instead (with ground truth)")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out-dir", default="src/cnn_segmentation/predictions")
    ap.add_argument("--show", action="store_true", help="also open each figure in a window")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LeNetUNet(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path, name in pick_samples(args):
        image, x, target = load_sample(image_path, mask_path, args.size)
        prediction = predict(model, x, device)
        out_path = out_dir / f"{name}.png"
        save_figure(image, prediction, target, name, out_path, args.show)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()