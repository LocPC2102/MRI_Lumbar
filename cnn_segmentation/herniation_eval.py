"""Geometric grading on a split, with ground-truth masks and predicted masks.

    python -m src.cnn_segmentation.herniation_eval --split test --top 30
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from src.cnn_segmentation.datasets.dataset import AxialSegmentationDataset
from src.cnn_segmentation.datasets.transforms import NUM_CLASSES, DualCompose, DualResize, DualToTensor
from src.cnn_segmentation.herniation_geometry import grade, overlay
from src.cnn_segmentation.models.lenet5_unet import LeNetUNet
from src.cnn_segmentation.predict import load_sample, predict


def summary(r):
    """-> (grade or 'rej' or '', ab/width or '')"""
    if r is None:
        return "", ""
    return ("rej" if r["flag"] else r["grade"]), round(float(r["ab_frac"]), 3)


def reasons(results):
    return dict(Counter("none" if r is None else (r["flag"] or "ok") for r in results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default="src/cnn_segmentation/models/checkpoints_4cls/lenet_unet_best.pt")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--top", type=int, default=30, help="save overlays of the top-N ab/w slices (ground-truth masks)")
    ap.add_argument("--out", default="src/cnn_segmentation/herniation/eval_split.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LeNetUNet(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    tf = DualCompose([DualResize(args.size), DualToTensor()])
    ds = AxialSegmentationDataset(f"DATA/processed/seg_{args.split}.csv", transform=tf)

    rows, res_gt, res_pr = [], {}, {}
    for i in range(len(ds)):
        x, y = ds[i]
        res_gt[i] = grade(y.numpy())
        res_pr[i] = grade(predict(model, x, device))
        g_gt, ab_gt = summary(res_gt[i])
        g_pr, ab_pr = summary(res_pr[i])
        r = ds.rows[i]
        rows.append([r["patient_id"], Path(r["image_path"]).stem, g_gt, g_pr, ab_gt, ab_pr])

    # slice position inside each patient (by file name order)
    by_patient = defaultdict(list)
    for k, row in enumerate(rows):
        by_patient[row[0]].append((row[1], k))
    rank = {k: j + 1 for items in by_patient.values() for j, (_, k) in enumerate(sorted(items))}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "image", "grade_gt", "grade_pred", "ab_w_gt", "ab_w_pred", "slice_rank"])
        w.writerows(row + [rank[k]] for k, row in enumerate(rows))

    valid = [r for r in rows if r[2] not in ("", "rej") and r[3] not in ("", "rej")]
    print(f"images {len(rows)} | usable in both {len(valid)} | rejected/none "
          f"gt {sum(r[2] in ('', 'rej') for r in rows)} pred {sum(r[3] in ('', 'rej') for r in rows)}")
    print("reject reasons, ground-truth masks:", reasons(res_gt.values()))
    print("reject reasons, predicted masks   :", reasons(res_pr.values()))
    print("grade distribution, ground-truth masks:", dict(sorted(Counter(r[2] for r in valid).items())))
    print("grade distribution, predicted masks   :", dict(sorted(Counter(r[3] for r in valid).items())))
    if valid:
        print(f"same grade: {np.mean([r[2] == r[3] for r in valid]):.1%} | "
              f"within 1 grade: {np.mean([abs(r[2] - r[3]) <= 1 for r in valid]):.1%} | "
              f"mean |ab/w difference|: {np.mean([abs(r[4] - r[5]) for r in valid]):.3f}")

    print("\nmean ab/w (ground-truth masks) by slice position inside the patient:")
    for j in sorted(set(rank.values())):
        v = [rows[k][4] for k in rank if rank[k] == j and rows[k][4] != "" and rows[k][2] != "rej"]
        if v:
            print(f"  slice {j}: n={len(v):3d}  mean {np.mean(v):+.3f}  max {np.max(v):+.3f}")

    top_dir = Path(args.out).parent / "top"
    top_dir.mkdir(exist_ok=True)
    ranked = sorted((k for k, r in res_gt.items() if r is not None and not r["flag"]),
                    key=lambda k: -res_gt[k]["ab_frac"])
    for n, k in enumerate(ranked[:args.top], 1):
        r = ds.rows[k]
        image, _, _ = load_sample(r["image_path"], r["mask_path"], args.size)
        name = f"{r['patient_id']}_{rows[k][1]}"
        overlay(image, res_gt[k], name, top_dir / f"{n:02d}_{name}.png")
    print(f"\nsaved {args.out} and top-{args.top} overlays in {top_dir}")


if __name__ == "__main__":
    main()