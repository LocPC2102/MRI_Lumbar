"""
Train the 5-ROI detector.

Run from D:\\test:
    python -m src.cnn.cnn_roi_detection.train

Checkpoint selection and early stopping use val_shift mean IoU (averaged over the 5 sites):
val is almost all centered, so it cannot tell a real detector from a "guess the usual place" model.
Outputs: outputs/best.pt, outputs/history.csv
"""

import csv
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as cfg
from .datasets.dataset import RoiDataset
from .eval.evaluate import evaluate
from .eval.evaluate import main as final_evaluation
from .loss.losses import detection_loss
from .models.detector import RoiDetector


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(split: str, shuffle: bool) -> DataLoader:
    return DataLoader(
        RoiDataset(cfg.MANIFEST_PATH, split),
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.DEVICE == "cuda",
    )


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total, count = 0.0, 0
    for images, boxes in loader:
        images, boxes = images.to(device), boxes.to(device)
        loss, parts = detection_loss(model(images), boxes)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += parts["loss"] * len(images)
        count += len(images)
    return total / count


def main() -> None:
    set_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader("train", shuffle=True)
    val_shift_loader = make_loader("val_shift", shuffle=False)
    val_loader = make_loader("val", shuffle=False)

    model = RoiDetector(in_channels=cfg.IN_CHANNELS, num_sites=cfg.NUM_SITES,
                        num_classes=cfg.NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {device} | params {n_params:,} | "
          f"train {len(train_loader.dataset)} | val_shift {len(val_shift_loader.dataset)} | "
          f"val {len(val_loader.dataset)}")

    fields = ["epoch", "lr", "train_loss", "val_shift_iou", "val_shift_center_err_px",
              "val_shift_cell_acc", "val_iou"]
    best_iou, best_epoch = -1.0, 0

    with (cfg.OUTPUT_DIR / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for epoch in range(1, cfg.EPOCHS + 1):
            lr = optimizer.param_groups[0]["lr"]
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            scheduler.step()

            shifted = evaluate(model, val_shift_loader, device)["metrics"]["ALL"]
            original = evaluate(model, val_loader, device)["metrics"]["ALL"]

            improved = shifted["mean_iou"] > best_iou
            if improved:
                best_iou, best_epoch = shifted["mean_iou"], epoch
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_shift_iou": best_iou}, cfg.CHECKPOINT_PATH)

            writer.writerow({
                "epoch": epoch, "lr": lr, "train_loss": train_loss,
                "val_shift_iou": shifted["mean_iou"],
                "val_shift_center_err_px": shifted["center_err_px"],
                "val_shift_cell_acc": shifted["cell_acc"],
                "val_iou": original["mean_iou"],
            })
            f.flush()

            print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | "
                  f"val_shift iou {shifted['mean_iou']:.4f} cell {shifted['cell_acc']:.3f} "
                  f"ctr {shifted['center_err_px']:.2f}px | val iou {original['mean_iou']:.4f}"
                  f"{'  *' if improved else ''}")

            if epoch - best_epoch >= cfg.EARLY_STOP_PATIENCE:
                print(f"Early stop: no val_shift IoU gain for {cfg.EARLY_STOP_PATIENCE} epochs.")
                break

    print(f"\nBest val_shift IoU {best_iou:.4f} at epoch {best_epoch} -> {cfg.CHECKPOINT_PATH}")
    final_evaluation()


if __name__ == "__main__":
    main()