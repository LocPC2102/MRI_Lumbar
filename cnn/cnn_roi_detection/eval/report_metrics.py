"""
Report for the 5-ROI detector: accuracy table, AP and site confusion matrix.

Place in src/cnn/cnn_roi_detection/eval/ and run from the project root:
    python -m src.cnn.cnn_roi_detection.eval.report_metrics
    python -m src.cnn.cnn_roi_detection.eval.report_metrics --ckpt src/cnn/cnn_roi_detection/outputs_256/best.pt

For each split (val, val_shift, test, test_shift) it prints and saves:
  accuracy   mean IoU, IoU@0.5, IoU@0.7, center / size error (px), cell accuracy,
             and the "always predict the train-mean box" baseline IoU
  AP         AP@0.5, AP@0.75, AP@0.5:0.95 per site (one box per site and image,
             ranked by its objectness score)
  confusion  rows = predicted site slot, columns = the GT site its box overlaps most
             (IoU >= --match-iou), last column "miss" = overlaps no GT box enough.
             A strong diagonal means each slot lands on its own site.

Saved in <outputs>/report/: report.csv, confusion_<split>.png, confusion_<split>.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import config as cfg
from ..datasets.dataset import RoiDataset
from ..loss.losses import decode_best_boxes, encode_targets
from ..models.detector import RoiDetector
from .evaluate import train_mean_box
from .metrics import constant_box_iou, per_item_metrics, summarize

SPLITS = ("val", "val_shift", "test", "test_shift")
COLUMNS = ("mean_iou", "iou@0.5", "iou@0.7", "center_err_px", "size_err_px", "cell_acc")
IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)


def make_loader(split: str) -> DataLoader:
    return DataLoader(RoiDataset(cfg.MANIFEST_PATH, split), batch_size=cfg.BATCH_SIZE,
                      shuffle=False, num_workers=cfg.NUM_WORKERS)


@torch.no_grad()
def predict(model, loader, device):
    """Best box of every site for the whole split: pred, gt [N, S, 4], score [N, S], hit [N, S]."""
    model.eval()
    preds, gts, scores, hits = [], [], [], []
    for images, gt in loader:
        images, gt = images.to(device), gt.to(device)
        raw = model(images)
        pred, score, gi, gj = decode_best_boxes(raw)
        t_gi, t_gj, valid, _, _ = encode_targets(gt.reshape(-1, 4), raw.shape[-1])
        hit = ((t_gi == gi.reshape(-1, 1)) & (t_gj == gj.reshape(-1, 1)) & valid).any(dim=1)
        preds.append(pred.cpu())
        gts.append(gt.cpu())
        scores.append(score.cpu())
        hits.append(hit.view(len(gt), cfg.NUM_SITES).cpu())
    return torch.cat(preds), torch.cat(gts), torch.cat(scores), torch.cat(hits)


# ------------------------------ AP ------------------------------------
def average_precision(scores: np.ndarray, is_tp: np.ndarray) -> float:
    """All-point AP. One prediction and one GT per image, so #GT = #predictions."""
    order = np.argsort(-scores)
    tp = is_tp[order].astype(float)
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(1.0 - tp)
    recall = cum_tp / len(tp)
    precision = cum_tp / (cum_tp + cum_fp)

    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def site_ap(iou: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "ap50": average_precision(score, iou >= 0.5),
        "ap75": average_precision(score, iou >= 0.75),
        "ap50_95": float(np.mean([average_precision(score, iou >= t) for t in IOU_THRESHOLDS])),
    }


# --------------------------- confusion --------------------------------
def pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b [N, S, 4] -> [N, S, S]: IoU of predicted box i against GT box j."""
    lt = torch.maximum(a[:, :, None, :2], b[:, None, :, :2])
    rb = torch.minimum(a[:, :, None, 2:], b[:, None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(dim=-1)[:, :, None]
    area_b = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(dim=-1)[:, None, :]
    return inter / (area_a + area_b - inter + 1e-7)


def confusion_counts(pred: torch.Tensor, gt: torch.Tensor, match_iou: float) -> np.ndarray:
    """[S, S + 1] counts: predicted slot i -> GT site with the highest IoU (or "miss")."""
    s = cfg.NUM_SITES
    best_iou, best_site = pairwise_iou(pred, gt).max(dim=2)             # [N, S]
    matched = torch.where(best_iou >= match_iou, best_site, torch.full_like(best_site, s))
    counts = np.zeros((s, s + 1), dtype=int)
    for slot in range(s):
        counts[slot] = np.bincount(matched[:, slot].numpy(), minlength=s + 1)
    return counts


def save_confusion(counts: np.ndarray, split: str, out_dir) -> None:
    names = [*cfg.SITES, "miss"]
    pct = 100.0 * counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)

    with (out_dir / f"confusion_{split}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pred_slot \\ gt_site", *names])
        for site, row in zip(cfg.SITES, counts):
            writer.writerow([site, *row.tolist()])

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    image = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(cfg.SITES)), cfg.SITES)
    ax.set_xlabel("GT site with highest IoU")
    ax.set_ylabel("Predicted site slot")
    ax.set_title(f"Site confusion ({split})")
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            ax.text(j, i, f"{pct[i, j]:.1f}%\n{counts[i, j]}", ha="center", va="center", fontsize=8,
                    color="white" if pct[i, j] > 55 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046, label="% of row")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{split}.png", dpi=150)
    plt.close(fig)


# ------------------------------ printing ------------------------------
def print_split(split, n_images, metrics, baseline, ap, counts):
    header = COLUMNS + ("baseline_iou", "ap50", "ap75", "ap50_95")
    print(f"\n[{split}] {n_images} images")
    print(f"  {'site':5s}" + "".join(f"{c:>15s}" for c in header))
    for site in ("ALL", *cfg.SITES):
        row = [metrics[site][c] for c in COLUMNS] + [baseline[site], ap[site]["ap50"],
                                                      ap[site]["ap75"], ap[site]["ap50_95"]]
        print(f"  {site:5s}" + "".join(f"{v:15.4f}" for v in row))

    names = [*cfg.SITES, "miss"]
    print("  confusion, % of each predicted slot (rows) over the GT site it overlaps (cols):")
    print("  " + " " * 6 + "".join(f"{n:>8s}" for n in names))
    for site, row in zip(cfg.SITES, counts):
        pct = 100.0 * row / max(row.sum(), 1)
        print(f"  {site:6s}" + "".join(f"{v:8.1f}" for v in pct))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(cfg.CHECKPOINT_PATH))
    ap.add_argument("--out", default=str(cfg.OUTPUT_DIR / "report"))
    ap.add_argument("--match-iou", type=float, default=0.3,
                    help="min IoU for a predicted box to count as overlapping a GT site")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.DEVICE)
    model = RoiDetector(in_channels=cfg.IN_CHANNELS, num_sites=cfg.NUM_SITES,
                        num_classes=cfg.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    print(f"[DET] {args.ckpt}")

    mean_box = train_mean_box(make_loader("train"))

    rows = []
    for split in SPLITS:
        pred, gt, score, hit = predict(model, make_loader(split), device)
        values = per_item_metrics(pred, gt, hit)
        metrics = summarize(values)
        baseline = constant_box_iou(mean_box, gt)

        iou, score_np = values["iou"].numpy(), score.numpy()
        ap_per_site = {s: site_ap(iou[:, i], score_np[:, i]) for i, s in enumerate(cfg.SITES)}
        ap_per_site["ALL"] = {k: float(np.mean([ap_per_site[s][k] for s in cfg.SITES]))
                              for k in ("ap50", "ap75", "ap50_95")}

        counts = confusion_counts(pred, gt, args.match_iou)
        save_confusion(counts, split, out_dir)
        print_split(split, len(gt), metrics, baseline, ap_per_site, counts)

        for site in ("ALL", *cfg.SITES):
            rows.append({"split": split, "site": site,
                         **{c: round(metrics[site][c], 4) for c in COLUMNS},
                         "baseline_iou": round(baseline[site], 4),
                         **{k: round(v, 4) for k, v in ap_per_site[site].items()}})

    with (out_dir / "report.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()