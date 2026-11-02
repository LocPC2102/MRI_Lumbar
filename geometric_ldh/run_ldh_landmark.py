import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.cnn.cnn_roi_detection.loss.losses import decode_best_boxes
from src.cnn.cnn_roi_detection.models.detector import RoiDetector
from src.cnn_segmentation.models.lenet5_unet import LeNetUNet
from src.geometric_ldh.anatomical_landmarks import AnatomicalLandmarkDetector
from src.geometric_ldh.landmark_ldh_engine import LandmarkLdhEngine, clean_segmentation

SITES = ("CCS", "LLS", "RLS", "LFS", "RFS")
IMAGE_SIZE = 256

SEG_CKPT = "src/cnn_segmentation/models/checkpoints_4cls/lenet_unet_best.pt"
DET_CKPT = "src/cnn/cnn_roi_detection/outputs_256/best.pt"

RANK = {
    0: lambda r: -r.gap_mm,
    1: lambda r: r.gap_mm,
    2: lambda r: -r.ab_mm,
    3: lambda r: -r.index_pct,
}

DETECTORS = [
    ("strict", AnatomicalLandmarkDetector()),
    ("relaxed", AnatomicalLandmarkDetector(
        min_component_area=10, smooth_kernel=7, peak_support_radius=3,
        peak_prominence_px=0.3, central_exclusion_ratio=0.04)),
]


# ============================================================
# Models and image paths
# ============================================================

def load_models(device, det_ckpt=DET_CKPT):
    seg = LeNetUNet(num_classes=4).to(device)
    seg.load_state_dict(torch.load(SEG_CKPT, map_location=device, weights_only=True))
    seg.eval()

    det = RoiDetector(in_channels=1, num_sites=5, num_classes=0).to(device)
    det.load_state_dict(torch.load(det_ckpt, map_location=device, weights_only=False)["model"])
    det.eval()
    print(f"[SEG] {SEG_CKPT}\n[DET] {det_ckpt}")
    return seg, det


def collect_aisslab_paths(root):
    paths, seen = [], set()
    for name in ("seg_train", "seg_val", "seg_test"):
        csv_path = root / f"DATA/processed/{name}.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        col = "image_path" if "image_path" in df.columns else df.columns[0]
        for p in df[col].dropna():
            p = root / p
            if p.is_file() and p not in seen:
                paths.append(p)
                seen.add(p)
    random.Random(42).shuffle(paths)
    return paths


def collect_lsma_paths(root):
    lsma = root / "DATA/raw/LSMA_PQR"
    base = lsma / "images_png" if (lsma / "images_png").is_dir() else lsma
    bad = ("sagittal", "sag", "localizer", "coronal", "qc")
    paths = sorted({
        p for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
        and not any(b in str(p).lower() for b in bad)
        and any(k in str(p).lower() for k in ("axial", "tra"))
    })
    random.Random(42).shuffle(paths)
    return paths


# ============================================================
# Inference + landmarks + geometry
# ============================================================

def infer(path, seg_model, det_model, device, det_size):
    pil = Image.open(path).convert("L")
    img = np.array(pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    x = torch.from_numpy(img)[None, None].to(device)

    if det_size == IMAGE_SIZE:
        x_det = x
    else:   # the detector has its own input size; its boxes are normalized
        small = np.array(pil.resize((det_size, det_size), Image.BILINEAR), dtype=np.float32) / 255.0
        x_det = torch.from_numpy(small)[None, None].to(device)

    with torch.no_grad():
        mask = seg_model((x - x.mean()) / (x.std() + 1e-6)).argmax(dim=1)[0].cpu().numpy()
        boxes_norm, _, _, _ = decode_best_boxes(det_model(x_det))
    boxes_px = boxes_norm[0].cpu().numpy() * IMAGE_SIZE

    mask = clean_segmentation(mask)          # remove stray pixels / far blobs before any geometry
    boxes = {SITES[i]: [float(v) for v in boxes_px[i]] for i in range(5)}
    return img, mask, boxes


def pe_top_points(mask):
    """Fallback A/B: top-most PE pixel on each side of the canal midline."""
    cc_x = np.where(mask == 2)[1]
    mid = float(cc_x.mean()) if len(cc_x) else mask.shape[1] / 2
    pts = []
    for side in (np.arange(mask.shape[1]) < mid, np.arange(mask.shape[1]) >= mid):
        ys, xs = np.where((mask == 1) & side[None, :])
        if len(ys) == 0:
            return None
        k = np.argsort(ys)[:5]
        pts.append((float(xs[k].mean()), float(ys[k].mean())))
    return pts[0], pts[1]


def find_landmarks(mask):
    """Returns (A, B, C, level, None) or (None, None, None, None, reason)."""
    first_reason = None
    for level, det in DETECTORS:
        lm = det.detect(mask)
        if lm.valid:
            return lm.left_facet, lm.right_facet, lm.spinous_tip, level, None
        first_reason = first_reason or lm.reason

    ab = pe_top_points(mask)
    if ab is not None and np.hypot(ab[1][0] - ab[0][0], ab[1][1] - ab[0][1]) >= 3:
        det = DETECTORS[0][1]
        C, _ = det._find_spinous_point(det._clean_mask(mask == det.pe_class), ab[0], ab[1])
        if C is not None and C[1] > max(ab[0][1], ab[1][1]):
            return ab[0], ab[1], C, "fallback", None
    return None, None, None, None, first_reason or "all landmark methods failed"


def analyze(path, models, engine, det_size):
    """Returns (img, mask, result, level, fail_reason)."""
    seg_model, det_model, device = models
    img, mask, boxes = infer(path, seg_model, det_model, device, det_size)
    A, B, C, level, reason = find_landmarks(mask)
    if A is None:
        return img, mask, None, None, f"landmarks: {reason}"

    name = path.stem if path.parent.name in ("images_png", "Images", "axial") else f"{path.parent.name}_{path.stem}"
    result = engine.analyze(mask, boxes, A, B, C, name)
    if result is None:
        return img, mask, None, level, "engine: no IVD / C behind AB / IVD outside range / ab>cd"
    return img, mask, result, level, None


def render(engine, img, mask, result, output_path):
    engine.render((img * 255).astype(np.uint8), mask, result, str(output_path))


def save_csv(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "grade", "zone_code", "zone", "side", "ab_mm", "gap_mm",
                    "cd_mm", "ab_over_cd", "landmarks"])
        for _, r, level in rows:
            w.writerow([r.image_name, r.grade_num, r.zone_code, r.zone_name, r.lateralization,
                        f"{r.ab_mm:.2f}", f"{r.gap_mm:.2f}", f"{r.cd_mm:.2f}",
                        f"{r.index_pct / 100:.3f}", level])


# ============================================================
# Modes
# ============================================================

def run_single(path, models, engine, out_root, det_size):
    img, mask, r, level, reason = analyze(path, models, engine, det_size)
    if r is None:
        print(f"\n[FAILED] {path.name}: {reason}")
        return
    out = out_root / f"grade_{r.grade_num}"
    out.mkdir(parents=True, exist_ok=True)
    out = out / f"{r.image_name}_{level}.png"
    render(engine, img, mask, r, out)
    save_csv([(path, r, level)], out_root / "results.csv")
    print(f"\n{r.image_name}: {r.grade} | {r.zone_name} ({r.lateralization}) | "
          f"ab={r.ab_mm:.2f}mm gap={r.gap_mm:.2f}mm cd={r.cd_mm:.2f}mm ab/cd={r.index_pct / 100:.3f} "
          f"| landmarks={level}\nSaved: {out}")


def run_dataset(paths, models, engine, out_root, per_grade, det_size):
    rows, fails, levels = [], Counter(), Counter()
    for i, p in enumerate(paths, 1):
        try:
            _, _, r, level, reason = analyze(p, models, engine, det_size)
        except Exception as exc:
            r, reason = None, f"exception: {exc}"
        if r is None:
            fails[reason] += 1
        else:
            rows.append((p, r, level))
            levels[level] += 1
        if i % 100 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)} processed")

    save_csv(rows, out_root / "results.csv")
    print(f"\nGraded {len(rows)} / {len(paths)} | landmark level: {dict(levels)}")
    for reason, n in fails.most_common():
        print(f"  skipped {n:5d}  {reason}")

    for g in range(4):
        group = sorted((x for x in rows if x[1].grade_num == g), key=lambda x: RANK[g](x[1]))
        g_dir = out_root / f"grade_{g}"
        g_dir.mkdir(parents=True, exist_ok=True)
        print(f"Grade {g}: {len(group)} images, rendering {min(per_grade, len(group))}")
        for rank, (p, _, _) in enumerate(group[:per_grade], 1):
            img, mask, r, level, _ = analyze(p, models, engine, det_size)
            if r is not None:
                render(engine, img, mask, r, g_dir / f"{rank:02d}_{r.image_name}_{level}.png")
    print("Output:", out_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="single axial image")
    ap.add_argument("--dataset", choices=["lsma", "aisslab"], default="aisslab")
    ap.add_argument("--per-grade", type=int, default=10)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--spacing", type=float, default=0.6, help="mm per pixel")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--det-ckpt", default=DET_CKPT, help="detection checkpoint")
    ap.add_argument("--det-size", type=int, default=256, help="detector input size (64 or 256)")
    args = ap.parse_args()

    root = Path.cwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = (*load_models(device, args.det_ckpt), device)
    engine = LandmarkLdhEngine(pixel_spacing_mm=args.spacing)
    out_root = Path(args.output_dir) if args.output_dir else root / f"src/geometric_ldh/outputs_{args.dataset}_landmark"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.input:
        run_single(Path(args.input), models, engine, out_root, args.det_size)
        return

    paths = collect_lsma_paths(root) if args.dataset == "lsma" else collect_aisslab_paths(root)
    if args.max_images:
        paths = paths[: args.max_images]
    if not paths:
        raise SystemExit(f"No images found for dataset '{args.dataset}'.")
    run_dataset(paths, models, engine, out_root, args.per_grade, args.det_size)


if __name__ == "__main__":
    main()