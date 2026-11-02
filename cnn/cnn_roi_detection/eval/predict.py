"""
Run the trained 5-ROI detector and save images with the predicted boxes drawn.

From D:\\test:

  python -m src.cnn.cnn_roi_detection.eval.predict
      Predicts test and test_shift images from the manifest.
      Colored boxes = prediction (CCS green, LLS orange, RLS blue, LFS yellow, RFS magenta),
      white boxes = ground truth. Options: --splits val_shift ...  --limit 0 (all)

  python -m src.cnn.cnn_roi_detection.eval.predict --input path\\to\\image_or_folder
      Predicts any grayscale axial slice(s), any size. Boxes are saved in the
      ORIGINAL image pixels.

Results: <project>/predictions/<split or custom>/  (annotated PNGs + predictions.csv)
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from .. import config as cfg
from ..datasets.dataset import RoiDataset
from ..loss.losses import box_iou, decode_best_boxes
from ..models.detector import RoiDetector

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
WHITE = (255, 255, 255)
SITE_COLORS = {  # BGR
    "CCS": (0, 255, 0), "LLS": (0, 165, 255), "RLS": (255, 128, 0),
    "LFS": (0, 255, 255), "RFS": (255, 0, 255),
}
COORDS = ("xmin", "ymin", "xmax", "ymax")
DISPLAY_SCALE = 8        # manifest images are 64x64, enlarge them for viewing


def load_model(device: torch.device) -> RoiDetector:
    model = RoiDetector(in_channels=cfg.IN_CHANNELS, num_sites=cfg.NUM_SITES,
                        num_classes=cfg.NUM_CLASSES).to(device)
    checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


@torch.no_grad()
def predict(model, images: torch.Tensor, device: torch.device):
    """images [B, 1, S, S] in [0, 1] -> normalized boxes [B, 5, 4], scores [B, 5]."""
    boxes, scores, _, _ = decode_best_boxes(model(images.to(device)))
    return boxes.cpu(), scores.cpu()


def draw_boxes(canvas: np.ndarray, boxes, colors) -> None:
    for box, color in zip(boxes, colors):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------- manifest images ---------------------------
def run_manifest(model, device, split: str, limit: int) -> None:
    dataset = RoiDataset(cfg.MANIFEST_PATH, split)
    n = len(dataset)
    picks = list(range(n)) if limit <= 0 or limit >= n else \
        np.linspace(0, n - 1, limit).astype(int).tolist()

    items = [dataset[i] for i in picks]
    images = torch.stack([image for image, _ in items])
    gt = torch.stack([boxes for _, boxes in items])
    pred, score = predict(model, images, device)
    iou = box_iou(pred.reshape(-1, 4), gt.reshape(-1, 4)).view(len(picks), cfg.NUM_SITES)

    out_dir = cfg.PREDICTIONS_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    side = cfg.IMAGE_SIZE * DISPLAY_SCALE
    colors = [SITE_COLORS[s] for s in cfg.SITES]
    rows = []

    for k, i in enumerate(picks):
        name = Path(dataset.rows[i]["image"]).stem
        gray = (images[k, 0].numpy() * 255).astype(np.uint8)
        canvas = cv2.cvtColor(cv2.resize(gray, (side, side), interpolation=cv2.INTER_NEAREST),
                              cv2.COLOR_GRAY2BGR)
        draw_boxes(canvas, gt[k] * side, [WHITE] * cfg.NUM_SITES)
        draw_boxes(canvas, pred[k] * side, colors)
        cv2.putText(canvas, f"mean IoU {iou[k].mean():.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, WHITE, 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{name}.png"), canvas)

        row = {"image": name, "mean_iou": round(iou[k].mean().item(), 4)}
        for j, site in enumerate(cfg.SITES):
            row[f"{site}_iou"] = round(iou[k, j].item(), 4)
            row[f"{site}_score"] = round(score[k, j].item(), 4)
        rows.append(row)

    write_csv(out_dir / "predictions.csv", rows)
    print(f"[{split}] {len(picks)} images | mean IoU {iou.mean():.4f} | saved to {out_dir}")


# ----------------------------- custom images ---------------------------
def run_custom(model, device, source: Path) -> None:
    files = [source] if source.is_file() else \
        sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No images found in {source}")

    out_dir = cfg.PREDICTIONS_DIR / "custom"
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = [SITE_COLORS[s] for s in cfg.SITES]
    rows = []

    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[WARN] cannot read {path}")
            continue
        h, w = image.shape
        # Same preprocessing as training: grayscale, resize (INTER_AREA), divide by 255.
        small = cv2.resize(image, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(small).float().div(255.0).view(1, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        boxes, score = predict(model, tensor, device)
        boxes = boxes[0] * torch.tensor([w, h, w, h])        # original pixels, [5, 4]

        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        draw_boxes(canvas, boxes, colors)
        cv2.imwrite(str(out_dir / f"{path.parent.name}__{path.stem}_pred.png"), canvas)

        row = {"image": str(path), "width": w, "height": h}
        for j, site in enumerate(cfg.SITES):
            row.update({f"{site}_{c}": round(v, 1) for c, v in zip(COORDS, boxes[j].tolist())})
            row[f"{site}_score"] = round(score[0, j].item(), 4)
        rows.append(row)

    write_csv(out_dir / "predictions.csv", rows)
    print(f"[custom] {len(rows)} images | saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the 5 ROI boxes and save annotated images.")
    parser.add_argument("--input", type=Path, default=None,
                        help="image file or folder; default = manifest splits")
    parser.add_argument("--splits", nargs="+", default=["test", "test_shift"],
                        help="manifest splits to predict (when --input is not given)")
    parser.add_argument("--limit", type=int, default=50,
                        help="max images per split, spread evenly (0 = all)")
    args = parser.parse_args()

    device = torch.device(cfg.DEVICE)
    model = load_model(device)

    if args.input is not None:
        run_custom(model, device, args.input)
    else:
        for split in args.splits:
            run_manifest(model, device, split, args.limit)


if __name__ == "__main__":
    main()