"""
Audit the 5-ROI dataset built by build_dataset.py.

Checks:
  1. Every image exists, is OUT_SIZE x OUT_SIZE, and all 5 boxes are valid.
  2. No patient appears in more than one split.
  3. The left-to-right order of the sites (RFS, RLS, CCS, LLS, LFS) is consistent.
Reports:
  4. Box size per site in output pixels (thin boxes limit how precisely they can be localized).
  5. How spread out the ROI-group center is per split (shift coverage on the detector grid).
  6. State counts per site (normal / stenosis / severe_stenosis).

Outputs (in dataset_5roi/):
  audit_center_heatmap.png   center heatmap of the 5-ROI group for every split
  audit_montage.png          16 random train images with their 5 boxes
"""

import csv
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parent / "dataset_5roi_128"
MANIFEST = DATASET_ROOT / "manifest.csv"

OUT_SIZE = 128     # must match build_dataset.py
GRID = 16           # detector grid (64 -> 8 after 3 pools)
MONTAGE_COUNT = 16
SEED = 0

SITES = ("CCS", "LLS", "RLS", "LFS", "RFS")            # slot order, same as build_dataset.py
LEFT_TO_RIGHT = ("RFS", "RLS", "CCS", "LLS", "LFS")    # layout seen in the XML files
COORDS = ("xmin", "ymin", "xmax", "ymax")
SPLITS = ("train", "val", "val_shift", "test", "test_shift")
COLORS = {  # BGR
    "CCS": (0, 255, 0), "LLS": (0, 165, 255), "RLS": (255, 128, 0),
    "LFS": (0, 255, 255), "RFS": (255, 0, 255),
}


def load_rows():
    with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def boxes_of(row) -> np.ndarray:
    """[5, 4] boxes in SITES order, output pixels."""
    return np.array([[float(row[f"{s}_{c}"]) for c in COORDS] for s in SITES])


# ------------------------------ checks --------------------------------
def check_files(rows, boxes):
    problems = []
    for r, b in zip(rows, boxes):
        img = cv2.imread(r["image"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            problems.append(f"cannot read image: {r['image']}")
            continue
        if img.shape != (OUT_SIZE, OUT_SIZE):
            problems.append(f"wrong size {img.shape}: {r['image']}")
        valid = ((b[:, 0] >= 0) & (b[:, 0] < b[:, 2]) & (b[:, 2] <= OUT_SIZE)
                 & (b[:, 1] >= 0) & (b[:, 1] < b[:, 3]) & (b[:, 3] <= OUT_SIZE))
        if not valid.all():
            bad = [s for s, ok in zip(SITES, valid) if not ok]
            problems.append(f"bad bbox {bad}: {r['image']}")
        if not Path(r["annotation"]).exists():
            problems.append(f"missing xml: {r['annotation']}")
    return problems


def check_leakage(rows):
    seen = {}
    for r in rows:
        seen.setdefault(r["patient"], set()).add(r["split"].removesuffix("_shift"))
    return {p: s for p, s in seen.items() if len(s) > 1}


def order_consistency(boxes) -> float:
    """Share of images whose 5 box centers follow LEFT_TO_RIGHT along x."""
    cx = (boxes[:, :, 0] + boxes[:, :, 2]) / 2
    ordered = cx[:, [SITES.index(s) for s in LEFT_TO_RIGHT]]
    return float((np.diff(ordered, axis=1) > 0).all(axis=1).mean())


# ------------------------------ reports -------------------------------
def size_report(boxes, splits):
    b = boxes[splits == "train"]
    w = b[:, :, 2] - b[:, :, 0]
    h = b[:, :, 3] - b[:, :, 1]
    print(f"\n[CHECK 4] box size per site in output pixels (train, {len(b)} images)")
    for i, s in enumerate(SITES):
        print(f"  {s}: w {w[:, i].min():.2f}-{w[:, i].max():.2f} (mean {w[:, i].mean():.2f})   "
              f"h {h[:, i].min():.2f}-{h[:, i].max():.2f} (mean {h[:, i].mean():.2f})")


def center_report(boxes, splits):
    """Per-split spread of the 5-ROI group center (center of the union box)."""
    union = np.stack([boxes[:, :, 0].min(1), boxes[:, :, 1].min(1),
                      boxes[:, :, 2].max(1), boxes[:, :, 3].max(1)], axis=1)
    print("\n[CHECK 5] ROI-group center spread")
    heatmaps = {}
    for split in SPLITS:
        u = union[splits == split]
        cx = (u[:, 0] + u[:, 2]) / 2 / OUT_SIZE
        cy = (u[:, 1] + u[:, 3]) / 2 / OUT_SIZE

        hist, _, _ = np.histogram2d(cy, cx, bins=GRID, range=[[0, 1], [0, 1]])
        heatmaps[split] = hist
        mid = GRID // 2
        center_share = hist[mid - 1:mid + 1, mid - 1:mid + 1].sum() / len(u)

        print(f"\n[{split}] {len(u)} images")
        print(f"  center x: mean {cx.mean():.3f}  std {cx.std():.3f}   "
              f"center y: mean {cy.mean():.3f}  std {cy.std():.3f}")
        print(f"  share of centers in the middle 2x2 cells: {center_share:.1%} "
              f"(uniform would be {4 / GRID ** 2:.1%})")
        print(f"  empty cells: {(hist == 0).sum()} / {GRID * GRID}")
        print(f"  {GRID}x{GRID} count grid (rows = y, cols = x):")
        for line in hist.astype(int):
            print("   " + " ".join(f"{v:4d}" for v in line))
    return heatmaps


def state_report(rows):
    print("\n[CHECK 6] states per site")
    for split in ("train", "val", "test"):
        sel = [r for r in rows if r["split"] == split]
        print(f"  [{split}] {len(sel)} images")
        for s in SITES:
            counts = Counter(r[f"{s}_state"] for r in sel)
            print(f"    {s}: " + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))


# ------------------------------ images --------------------------------
def save_heatmaps(heatmaps):
    tiles = []
    for split, hist in heatmaps.items():
        norm = (hist / max(hist.max(), 1) * 255).astype(np.uint8)
        big = cv2.resize(norm, (256, 256), interpolation=cv2.INTER_NEAREST)
        tile = cv2.applyColorMap(big, cv2.COLORMAP_VIRIDIS)
        cv2.putText(tile, split, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    cv2.imwrite(str(DATASET_ROOT / "audit_center_heatmap.png"), np.hstack(tiles))


def save_montage(rows, boxes, splits):
    rng = random.Random(SEED)
    idx = np.flatnonzero(splits == "train").tolist()
    picks = rng.sample(idx, min(MONTAGE_COUNT, len(idx)))
    scale = 4
    tiles = []
    for i in picks:
        img = cv2.imread(rows[i]["image"], cv2.IMREAD_GRAYSCALE)
        big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        tile = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        for s, box in zip(SITES, boxes[i]):
            x1, y1, x2, y2 = [int(round(v * scale)) for v in box]
            cv2.rectangle(tile, (x1, y1), (x2, y2), COLORS[s], 1)
        cv2.putText(tile, rows[i]["transform"], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 4:  # pad so the grid has full rows
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(DATASET_ROOT / "audit_montage.png"), grid)


# ------------------------------- main ---------------------------------
def main():
    rows = load_rows()
    boxes = np.stack([boxes_of(r) for r in rows])          # [N, 5, 4]
    splits = np.array([r["split"] for r in rows])
    print(f"[INFO] manifest rows: {len(rows)}")

    problems = check_files(rows, boxes)
    print(f"\n[CHECK 1] files / size / boxes : {'OK' if not problems else f'{len(problems)} PROBLEMS'}")
    for p in problems[:10]:
        print("  -", p)

    leaks = check_leakage(rows)
    print(f"[CHECK 2] patient leakage      : {'OK' if not leaks else f'{len(leaks)} PATIENTS IN >1 SPLIT'}")
    for p, s in list(leaks.items())[:10]:
        print(f"  - patient {p}: {sorted(s)}")

    print(f"[CHECK 3] left-to-right order  : {order_consistency(boxes):.1%} of images follow "
          f"{' < '.join(LEFT_TO_RIGHT)}")

    size_report(boxes, splits)
    heatmaps = center_report(boxes, splits)
    state_report(rows)
    save_heatmaps(heatmaps)
    save_montage(rows, boxes, splits)

    print(f"\nSaved: {DATASET_ROOT / 'audit_center_heatmap.png'}")
    print(f"Saved: {DATASET_ROOT / 'audit_montage.png'}")
    print("Expect: train and *_shift spread over the grid (middle-2x2 share well under 40%, few empty cells);")
    print("        val/test (originals) stay concentrated in the middle.")


if __name__ == "__main__":
    main()