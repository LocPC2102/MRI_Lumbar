"""Geometric herniation grading on axial slices (after Radiol Med 2025, doi 10.1007/s11547-025-01996-y).

    python -m src.cnn_segmentation.herniation_geometry --input tesst
    python -m src.cnn_segmentation.herniation_geometry --input tesst --boxes boxes.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from src.cnn_segmentation.datasets.transforms import NUM_CLASSES
from src.cnn_segmentation.models.lenet5_unet import LeNetUNet
from src.cnn_segmentation.predict import IMAGE_SUFFIXES, load_sample, predict

PE, CC, IVD = 1, 2, 3
TOL = 0.06        # grade 0 when the disc stays farther than this (fraction of disc width) behind the line
EPS = 0.03        # dead zone: excess below this is mask noise, not herniation (~3 px)
RATIO = 0.3       # sagittal diameter index threshold from the paper
CANAL = 0.4       # canal depth behind the facet line as a fraction of disc width (assumption)
MAX_TILT = 15     # degrees, facet line vs lateral axis; larger -> probably not an axial slice
MAX_AB = 0.3      # |ab / disc width| above this -> broken mask, reject


def largest(mask):
    comp, n = ndimage.label(mask)
    return comp == np.bincount(comp.ravel())[1:].argmax() + 1 if n else mask


def xy(mask):
    """Mask -> (N, 2) float array of (x, y) pixel coordinates."""
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1).astype(float)


def frame(labels):
    """Spine frame: origin at CC centre, a = posterior direction, l = lateral direction."""
    cc = xy(labels == CC)
    if len(cc) < 10:
        return None                                    # no usable canal -> cannot orient
    c = cc.mean(0)
    a = c - xy(largest(labels == IVD)).mean(0)
    norm = np.linalg.norm(a)
    if norm < 1e-6:
        return None
    a = a / norm
    return c, a, np.array([a[1], -a[0]])


def facet_line(P, c, a, l):
    """A, B = highest PE pixel on each side; on a flat top, the most lateral pixel of that row."""
    t, s = (P - c) @ a, (P - c) @ l
    pts = []
    for sign in (-1, 1):
        side = np.flatnonzero(sign * s > 0)
        if len(side) < 5:
            return None
        top = side[t[side] <= t[side].min() + 1.0]     # highest row, 1 px tolerance
        pts.append(P[top[np.argmax(sign * s[top])]])   # most lateral pixel in it
    return pts


def grade(labels):
    """labels: [H, W] class ids (1 PE, 2 CC, 3 IVD). Returns a dict, or None if no disc/canal/PE."""
    if not (labels == IVD).any() or not (labels == PE).any():
        return None
    fr = frame(labels)
    if fr is None:
        return None
    c, a, l = fr
    ivd = largest(labels == IVD)
    ivd_xy, pe_xy = xy(ivd), xy(labels == PE)

    width = np.ptp((ivd_xy - c) @ l)
    line = facet_line(pe_xy, c, a, l)
    if line is None:
        return None
    A, B = line
    length = np.linalg.norm(B - A)
    if length < 1e-6:
        return None
    u = (B - A) / length
    n = np.array([-u[1], u[0]])
    if n @ a < 0:
        n = -n                                          # make n point posterior

    d = (ivd_xy - A) @ n                                # depth behind the facet line
    p = (ivd_xy - A) @ u                                # position along the line
    inner = (p > 0.15 * length) & (p < 0.85 * length)   # ignore the disc corners at the line ends
    if not inner.any():
        return None
    k = np.flatnonzero(inner)[d[inner].argmax()]
    ab = d[k]                                           # > 0: disc goes past the facet line
    frac = ab / width

    tilt = np.degrees(np.arccos(np.clip(u @ l, -1, 1)))
    flag = ""
    if tilt > MAX_TILT:
        flag = "tilt"
    elif not 0.5 < length / width < 1.5:
        flag = "line_length"
    elif abs(frac) > MAX_AB:
        flag = "implausible"

    ratio = np.nan
    if frac < -TOL:
        g = 0
    elif frac <= EPS:
        g = 1
    else:
        ratio = ab / (CANAL * width)
        g = 2 if ratio <= RATIO else 3
    return dict(grade=g, flag=flag, ab_frac=frac, ratio=ratio, tilt=tilt,
                apex=tuple(ivd_xy[k]), A=A, B=B, ivd=ivd,
                pe=labels == PE, cc=labels == CC)


def zone(apex_x, boxes):
    """Detection box (CCS/LLS/RLS/LFS/RFS) whose centre is closest to the apex in x."""
    return min(boxes, key=lambda s: abs((boxes[s][0] + boxes[s][2]) / 2 - apex_x)) if boxes else ""


def read_boxes(path):
    """CSV columns: image (stem), site, x1, y1, x2, y2, in the 256x256 space."""
    boxes = {}
    if path:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                boxes.setdefault(r["image"], {})[r["site"]] = tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
    return boxes


def overlay(image, r, name, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray")
    ax.contour(r["ivd"], levels=[0.5], colors="yellow", linewidths=1)      # IVD
    ax.contour(r["pe"], levels=[0.5], colors="orange", linewidths=0.8)     # PE
    ax.contour(r["cc"], levels=[0.5], colors="lime", linewidths=0.8)       # CC
    ax.plot(*zip(r["A"], r["B"]), "r-", lw=1)                              # facet line
    ax.plot(*r["apex"], "co", ms=3)                                        # deepest disc point
    shown = f"REJECT:{r['flag']}" if r["flag"] else f"grade {r['grade']}"
    ax.set_title(f"{name}  {shown}  ab/w {r['ab_frac']:.2f}  tilt {r['tilt']:.0f}°", fontsize=8)
    ax.axis("off")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="src/cnn_segmentation/models/checkpoints_4cls/lenet_unet_best.pt")
    ap.add_argument("--input", default="tesst")
    ap.add_argument("--boxes", default=None)
    ap.add_argument("--out-dir", default="src/cnn_segmentation/herniation")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LeNetUNet(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    boxes = read_boxes(args.boxes)
    src = Path(args.input)
    paths = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "herniation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "grade", "ab_over_width", "ab_over_canal", "tilt_deg", "flag", "zone"])
        for p in paths:
            image, x, _ = load_sample(p, None, args.size)
            r = grade(predict(model, x, device))
            if r is None:
                w.writerow([p.stem, "", "", "", "", "no_disc_or_facet", ""])
                continue
            ratio = "" if np.isnan(r["ratio"]) else f"{r['ratio']:.3f}"
            g = "" if r["flag"] else r["grade"]
            w.writerow([p.stem, g, f"{r['ab_frac']:.3f}", ratio, f"{r['tilt']:.0f}", r["flag"],
                        zone(r["apex"][0], boxes.get(p.stem))])
            overlay(image, r, p.stem, out / f"{p.stem}.png")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()