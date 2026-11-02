"""
Script to evaluate LeNet U-Net (lenet_unet_best.pt) across train, val, and test splits,
generate confusion matrix visualizations, and export comprehensive metrics & report.md.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
from src.cnn_segmentation.eval.report_metrics import load_checkpoint, evaluate


def evaluate_split(model, split, batch_size=32, size=256, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path.cwd()
    csv_path = project_root / f"DATA/processed/seg_{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing split CSV: {csv_path}")

    transform = DualCompose([DualResize(size), DualToTensor()])
    dataset = AxialSegmentationDataset(csv_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    res = evaluate(model, loader, device, NUM_CLASSES)
    res["split"] = split
    res["num_samples"] = len(dataset)
    return res


def plot_confusion_matrices(splits_results, output_path, output_test_path):
    # Plot all 3 splits in one figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    class_names = list(CLASS_NAMES)

    for ax, split in zip(axes, ["train", "val", "test"]):
        cm = np.array(splits_results[split]["confusion_matrix"], dtype=np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
        ax.set_title(f"Confusion Matrix: {split.upper()} Split\n(Accuracy: {splits_results[split]['pixel_accuracy']*100:.2f}%)", fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=11)
        ax.set_yticklabels(class_names, fontsize=11)
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight="semibold")
        ax.set_ylabel("True Class", fontsize=11, fontweight="semibold")

        # Annotate
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                count = int(cm[i, j])
                pct = cm_norm[i, j]
                text_color = "white" if pct > 50 else "black"
                if count >= 1_000_000:
                    cnt_str = f"{count/1_000_000:.1f}M"
                elif count >= 1_000:
                    cnt_str = f"{count/1_000:.1f}k"
                else:
                    cnt_str = f"{count}"
                ax.text(
                    j, i, f"{pct:.1f}%\n({cnt_str})",
                    ha="center", va="center",
                    color=text_color, fontsize=9, fontweight="medium"
                )

    plt.tight_layout()
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Recall / Pixel %")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved 3-split confusion matrix figure: {output_path}")

    # Also plot standalone Test split confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cm = np.array(splits_results["test"]["confusion_matrix"], dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    ax.set_title(f"Test Split Confusion Matrix (LeNet U-Net)\nPixel Acc: {splits_results['test']['pixel_accuracy']*100:.2f}%, mIoU: {splits_results['test']['mean_iou']:.4f}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="semibold")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = int(cm[i, j])
            pct = cm_norm[i, j]
            text_color = "white" if pct > 50 else "black"
            if count >= 1_000_000:
                cnt_str = f"{count/1_000_000:.2f}M"
            elif count >= 1_000:
                cnt_str = f"{count/1_000:.1f}k"
            else:
                cnt_str = f"{count}"
            ax.text(
                j, i, f"{pct:.2f}%\n({cnt_str})",
                ha="center", va="center",
                color=text_color, fontsize=10, fontweight="medium"
            )

    fig.colorbar(im, ax=ax, shrink=0.8, label="Recall / Pixel %")
    plt.tight_layout()
    plt.savefig(output_test_path, bbox_inches="tight")
    plt.close()
    print(f"Saved test confusion matrix figure: {output_test_path}")


def main():
    project_root = Path.cwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_path = project_root / "src/cnn_segmentation/models/checkpoints_4cls/lenet_unet_best.pt"
    if not ckpt_path.is_file():
        ckpt_path = project_root / "src/cnn_segmentation/models/checkpoints/lenet_unet_best.pt"

    print(f"Loading checkpoint from: {ckpt_path}")
    model = LeNetUNet(num_classes=NUM_CLASSES).to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    splits = ["train", "val", "test"]
    all_results = {}

    for split in splits:
        print(f"\n--- Evaluating {split.upper()} split ---")
        res = evaluate_split(model, split, batch_size=32, device=device)
        all_results[split] = res
        print(f"Pixel Accuracy : {res['pixel_accuracy']:.4f} ({res['pixel_accuracy']*100:.2f}%)")
        print(f"Mean IoU       : {res['mean_iou']:.4f}")
        print(f"Mean Dice      : {res['mean_dice']:.4f}")

    # Save all results to JSON
    json_out = project_root / "src/cnn_segmentation/eval/all_splits_metrics.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved all metrics to: {json_out}")

    # Plot confusion matrix
    cm_all_path = project_root / "src/cnn_segmentation/eval/confusion_matrices_all_splits.png"
    cm_test_path = project_root / "src/cnn_segmentation/eval/confusion_matrix_test.png"
    plot_confusion_matrices(all_results, cm_all_path, cm_test_path)


if __name__ == "__main__":
    main()
