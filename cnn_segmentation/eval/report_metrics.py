"""
Evaluate the saved LeNet U-Net segmentation checkpoint on AISSLab Axial.

Run from the project root:
    python -m src.cnn_segmentation.eval.report_metrics

Optional:
    python -m src.cnn_segmentation.eval.report_metrics --split test
    python -m src.cnn_segmentation.eval.report_metrics --batch-size 16

The script does not train or modify the checkpoint.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.cnn_segmentation.datasets.dataset import AxialSegmentationDataset
from src.cnn_segmentation.datasets.transforms import (
    CLASS_NAMES,
    NUM_CLASSES,
    DualCompose,
    DualResize,
    DualToTensor,
)
from src.cnn_segmentation.models.lenet5_unet import LeNetUNet


def load_checkpoint(model, checkpoint_path: Path, device: torch.device) -> None:
    """Load either a plain state_dict or a common wrapped checkpoint."""
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch releases.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint format: {type(checkpoint).__name__}"
        )

    # Handle checkpoints saved from torch.nn.DataParallel.
    if checkpoint and all(k.startswith("module.") for k in checkpoint):
        checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint, strict=True)


@torch.inference_mode()
def evaluate(model, loader, device, num_classes):
    """Calculate metrics from a global pixel confusion matrix."""
    model.eval()
    confusion = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64, device=device
    )
    num_images = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).long()

        logits = model(images)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if logits.ndim != 4 or logits.shape[1] != num_classes:
            raise ValueError(
                f"Expected logits [B, {num_classes}, H, W], "
                f"received {tuple(logits.shape)}"
            )

        predictions = logits.argmax(dim=1)

        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]

        if predictions.shape != masks.shape:
            raise ValueError(
                f"Prediction shape {tuple(predictions.shape)} does not match "
                f"mask shape {tuple(masks.shape)}. Check transforms and mask format."
            )

        # Ignore mask pixels outside the configured class range.
        valid = (masks >= 0) & (masks < num_classes)
        encoded = masks[valid] * num_classes + predictions[valid]
        confusion += torch.bincount(
            encoded, minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)

        num_images += images.shape[0]

    cm = confusion.to(torch.float64)
    tp = cm.diag()
    gt_pixels = cm.sum(dim=1)
    pred_pixels = cm.sum(dim=0)
    union = gt_pixels + pred_pixels - tp

    # Undefined classes (no ground-truth and no predictions) are excluded
    # from macro averages, matching the usual nanmean convention.
    iou = torch.full_like(tp, float("nan"))
    iou[union > 0] = tp[union > 0] / union[union > 0]

    dice_denom = gt_pixels + pred_pixels
    dice = torch.full_like(tp, float("nan"))
    dice[dice_denom > 0] = (
        2 * tp[dice_denom > 0] / dice_denom[dice_denom > 0]
    )

    class_accuracy = torch.full_like(tp, float("nan"))
    class_accuracy[gt_pixels > 0] = (
        tp[gt_pixels > 0] / gt_pixels[gt_pixels > 0]
    )

    total_pixels = cm.sum()
    pixel_accuracy = (tp.sum() / total_pixels).item() if total_pixels else float("nan")

    def nanmean(values):
        valid_values = values[~torch.isnan(values)]
        return valid_values.mean().item() if valid_values.numel() else float("nan")

    results = {
        "num_images": num_images,
        "num_pixels_evaluated": int(total_pixels.item()),
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": nanmean(iou),
        "mean_dice": nanmean(dice),
        "per_class": {},
        "confusion_matrix": confusion.cpu().tolist(),
        "confusion_matrix_note": "Rows=true class, columns=predicted class",
    }

    for i, name in enumerate(CLASS_NAMES):
        results["per_class"][str(name)] = {
            "iou": iou[i].item(),
            "dice": dice[i].item(),
            "pixel_accuracy_recall": class_accuracy[i].item(),
            "ground_truth_pixels": int(gt_pixels[i].item()),
            "predicted_pixels": int(pred_pixels[i].item()),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Report segmentation metrics for lenet_unet_best.pt"
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--checkpoint",
        default="src/cnn_segmentation/models/checkpoints_4cls/lenet_unet_best.pt",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--output",
        default="src/cnn_segmentation/eval/segmentation_metrics.json",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    checkpoint_path = project_root / args.checkpoint
    csv_path = project_root / f"DATA/processed/seg_{args.split}.csv"
    output_path = project_root / args.output

    if not checkpoint_path.is_file():
        # Fallback to alternate checkpoint path if default was not found
        alt_path = project_root / "src/cnn_segmentation/models/checkpoints/lenet_unet_best.pt"
        if alt_path.is_file():
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset split CSV not found: {csv_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = DualCompose([DualResize(args.size), DualToTensor()])
    dataset = AxialSegmentationDataset(csv_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = LeNetUNet(num_classes=NUM_CLASSES).to(device)
    load_checkpoint(model, checkpoint_path, device)

    print("=" * 64)
    print("LeNet U-Net Segmentation Evaluation")
    print("=" * 64)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Split      : {args.split}")
    print(f"CSV        : {csv_path}")
    print(f"Images     : {len(dataset)}")
    print(f"Input size : {args.size} x {args.size}")
    print(f"Device     : {device}")

    results = evaluate(model, loader, device, NUM_CLASSES)
    results["model"] = "LeNetUNet"
    results["checkpoint"] = str(args.checkpoint)
    results["split"] = args.split
    results["input_size"] = [args.size, args.size]
    results["device"] = str(device)
    results["class_names"] = list(CLASS_NAMES)

    print("\nOVERALL")
    print(f"Pixel Accuracy : {results['pixel_accuracy']:.4f} "
          f"({results['pixel_accuracy'] * 100:.2f}%)")
    print(f"Mean IoU       : {results['mean_iou']:.4f}")
    print(f"Mean Dice      : {results['mean_dice']:.4f}")

    print("\nPER-CLASS METRICS")
    print(f"{'Class':<14} {'IoU':>10} {'Dice':>10} {'Pixel recall':>14}")
    print("-" * 52)
    for name, metrics in results["per_class"].items():
        print(
            f"{name:<14} "
            f"{metrics['iou']:>10.4f} "
            f"{metrics['dice']:>10.4f} "
            f"{metrics['pixel_accuracy_recall']:>13.2%}"
            if metrics["pixel_accuracy_recall"] == metrics["pixel_accuracy_recall"]
            else f"{name:<14} {metrics['iou']:>10.4f} {metrics['dice']:>10.4f} {'N/A':>14}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nJSON report saved to: {output_path}")


if __name__ == "__main__":
    main()
