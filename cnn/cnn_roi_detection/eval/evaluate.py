"""
Evaluate the trained 5-ROI detector.

Import `evaluate` from train.py, or run this file to score the best checkpoint
on val, val_shift, test and test_shift:
    python -m src.cnn.cnn_roi_detection.eval.evaluate

cell_acc = share of predictions whose cell is one of the site's positive cells
(the cell holding the true center or its two nearest neighbours).
"""

import torch
from torch.utils.data import DataLoader

from .. import config as cfg
from ..datasets.dataset import RoiDataset
from ..loss.losses import decode_best_boxes, detection_loss, encode_targets
from ..models.detector import RoiDetector
from .metrics import constant_box_iou, per_item_metrics, summarize

COLUMNS = ("mean_iou", "iou@0.5", "iou@0.7", "coverage", "center_err_px", "size_err_px", "cell_acc")


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Returns {"loss": float, "metrics": {"ALL": {...}, "CCS": {...}, ...}}."""
    model.eval()
    preds, gts, hits, losses = [], [], [], []

    for images, gt in loader:
        images, gt = images.to(device), gt.to(device)
        raw = model(images)

        loss, _ = detection_loss(raw, gt)
        pred, _, gi, gj = decode_best_boxes(raw)
        t_gi, t_gj, valid, _, _ = encode_targets(gt.reshape(-1, 4), raw.shape[-1])
        hit = ((t_gi == gi.reshape(-1, 1)) & (t_gj == gj.reshape(-1, 1)) & valid).any(dim=1)

        losses.append(loss.detach().expand(len(gt)))
        preds.append(pred)
        gts.append(gt)
        hits.append(hit.view(len(gt), cfg.NUM_SITES))

    values = per_item_metrics(torch.cat(preds), torch.cat(gts), torch.cat(hits))
    return {"loss": torch.cat(losses).mean().item(), "metrics": summarize(values)}


@torch.no_grad()
def train_mean_box(train_loader) -> torch.Tensor:
    """Average box of every site over the train set -> [S, 4]."""
    return torch.cat([gt for _, gt in train_loader]).mean(dim=0)


def baseline_for(mean_box: torch.Tensor, loader) -> dict[str, float]:
    gt = torch.cat([boxes for _, boxes in loader])
    return constant_box_iou(mean_box, gt)


def print_report(name: str, n_images: int, result: dict, baseline: dict) -> None:
    print(f"\n[{name}] {n_images} images | loss {result['loss']:.4f}")
    print(f"  {'site':5s}" + "".join(f"{c:>15s}" for c in COLUMNS) + f"{'baseline_iou':>15s}")
    for site in ("ALL", *cfg.SITES):
        row = result["metrics"][site]
        print(f"  {site:5s}" + "".join(f"{row[c]:15.4f}" for c in COLUMNS) + f"{baseline[site]:15.4f}")


def main() -> None:
    device = torch.device(cfg.DEVICE)
    model = RoiDetector(in_channels=cfg.IN_CHANNELS, num_sites=cfg.NUM_SITES,
                        num_classes=cfg.NUM_CLASSES).to(device)
    checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])

    def make_loader(split: str) -> DataLoader:
        return DataLoader(RoiDataset(cfg.MANIFEST_PATH, split), batch_size=cfg.BATCH_SIZE,
                          shuffle=False, num_workers=cfg.NUM_WORKERS)

    mean_box = train_mean_box(make_loader("train"))
    for split in ("val", "val_shift", "test", "test_shift"):
        loader = make_loader(split)
        result = evaluate(model, loader, device)
        print_report(split, len(loader.dataset), result, baseline_for(mean_box, loader))


if __name__ == "__main__":
    main()