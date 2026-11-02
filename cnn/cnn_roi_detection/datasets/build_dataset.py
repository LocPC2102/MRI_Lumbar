"""
Build the axial 5-ROI detection dataset (64x64, 1 channel) from AISSLab Axial VOC XML.

Source:
  D:\\test\\DATA\\raw\\AISSLab_Axial\\Localization_and_Severity_Classification\\0001..0500\\
  One folder = one patient (PNG images + same-stem XML).
  Each XML has 5 objects named <site>_<state>: CCS, LLS, RLS, LFS, RFS
  (state = normal / stenosis / severe_stenosis).

Output:
  D:\\test\\src\\cnn\\cnn_roi_detection\\datasets\\dataset_5roi\\
    images\\       64x64 PNG
    annotations\\  VOC-style XML with the 5 boxes (float coordinates, output pixels)
    manifest.csv   one row per image: split, transform, 5 boxes + 5 states
    splits.csv     image-level split assignment
    previews\\     images with the 5 boxes drawn (4x enlarged)

Splits (assigned per PATIENT, so no patient appears in two splits):
  train        TRAIN_COPIES independent random views per source image
  val / test   the original image (real-data distribution, ROIs sit near the center)
  val_shift /  the same source image with one random SHIFTED view (never "original"),
  test_shift   used to check the model really localizes instead of guessing the center

Rules:
- An image is kept only if its XML has exactly one valid box for each of the 5 sites.
- ONE affine transform (zoom + translation) is applied to the image and all 5 boxes together.
- No rotation: it would inflate these thin boxes (axis-aligned box of a rotated box).
"""

from __future__ import annotations

import csv
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ----------------------------- CONFIG ---------------------------------
SOURCE_ROOT = Path(
    r"D:\test\DATA\raw\AISSLab_Axial\Localization_and_Severity_Classification"
)
OUTPUT_ROOT = Path(r"D:\test\src\cnn\cnn_roi_detection\datasets\dataset_5roi_128")

SITES = ("CCS", "LLS", "RLS", "LFS", "RFS")   # slot order, must match dataset.py
COORDS = ("xmin", "ymin", "xmax", "ymax")

OUT_SIZE = 128           # final image size (OUT_SIZE x OUT_SIZE, 1 channel)

# Deletes OUTPUT_ROOT before writing. Keep True when regenerating so old
# files do not mix with the new ones.
CLEAN_OUTPUT_FIRST = True

SAVE_PREVIEWS = True
PREVIEW_SCALE = 2        # 64x64 is tiny, enlarge previews for spot-checking

SEED = 42

# Split ratios are applied to PATIENTS, not images.
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10         # remainder goes to test

# Each train source image gets this many independent random views.
TRAIN_COPIES = 4

# +/-35% of 64 px = +/-22 px = about +/-2.8 grid cells, so the ROI group covers the 8x8 grid.
TRANSLATE_FRACTION = 0.35
ZOOM_RANGE = (0.78, 1.25)   # <1 zoom out, >1 zoom in

# One random transform per view; weights sum to 1.
# Zoom acts around the image center and barely moves the ROIs,
# so most of the weight goes to the transforms that include translation.
TRANSFORM_WEIGHTS = {
    "original": 0.10,
    "translation": 0.45,
    "zoom": 0.05,
    "zoom_translation": 0.40,
}

# Reject a view if any box keeps less than this fraction of its area inside the image.
MIN_RETAINED_BOX_FRACTION = 0.9

VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "val_shift", "test", "test_shift")

PREVIEW_COLORS = {  # BGR
    "CCS": (0, 255, 0), "LLS": (0, 165, 255), "RLS": (255, 128, 0),
    "LFS": (0, 255, 255), "RFS": (255, 0, 255),
}
# ----------------------------------------------------------------------

Box = tuple[float, float, float, float]
Boxes = dict[str, Box]   # site -> box


@dataclass
class Annotation:
    width: int
    height: int
    boxes: Boxes
    states: dict[str, str]


# ----------------------------- XML / BOX ------------------------------
def _tag_text(parent: ET.Element, tag: str, default: str = "") -> str:
    node = parent.find(tag)
    return node.text.strip() if node is not None and node.text else default


def clip_box(box: Box, width: int, height: int) -> Optional[Box]:
    x1, y1, x2, y2 = box
    x1, x2 = np.clip([x1, x2], 0, width - 1)
    y1, y2 = np.clip([y1, y2], 0, height - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return float(x1), float(y1), float(x2), float(y2)


def parse_annotation(xml_path: Path) -> tuple[Optional[Annotation], str]:
    """Read the 5 site boxes. Returns (annotation, "") or (None, reason it was rejected)."""
    try:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        width = int(_tag_text(size, "width", "0")) if size is not None else 0
        height = int(_tag_text(size, "height", "0")) if size is not None else 0
    except Exception as exc:
        print(f"[WARN] Could not parse XML {xml_path}: {exc}")
        return None, "bad_xml"
    if width <= 0 or height <= 0:
        return None, "bad_size"

    boxes: Boxes = {}
    states: dict[str, str] = {}
    for obj in root.findall("object"):
        site, _, state = _tag_text(obj, "name").partition("_")   # "LLS_normal" -> LLS, normal
        bnd = obj.find("bndbox")
        if site not in SITES or bnd is None:
            continue
        if site in boxes:
            return None, "duplicate_site"
        try:
            raw = tuple(float(_tag_text(bnd, t)) for t in COORDS)
        except ValueError:
            return None, "bad_box"
        box = clip_box(raw, width, height)
        if box is None:
            return None, "bad_box"
        boxes[site] = box
        states[site] = state or "unknown"

    if any(site not in boxes for site in SITES):
        return None, "missing_site"
    return Annotation(width, height, boxes, states), ""


# --------------------------- AUGMENTATION -----------------------------
def make_affine(width: int, height: int, scale: float, tx: float, ty: float) -> np.ndarray:
    """Zoom about the image center, then translate."""
    cx, cy = width / 2.0, height / 2.0
    return np.array(
        [[scale, 0.0, cx - scale * cx + tx],
         [0.0, scale, cy - scale * cy + ty]],
        dtype=np.float32,
    )


def sample_transform(kind: str, width: int, height: int, rng: random.Random) -> np.ndarray:
    scale, tx, ty = 1.0, 0.0, 0.0
    if "translation" in kind:
        tx = rng.uniform(-TRANSLATE_FRACTION, TRANSLATE_FRACTION) * width
        ty = rng.uniform(-TRANSLATE_FRACTION, TRANSLATE_FRACTION) * height
    if "zoom" in kind:
        scale = rng.uniform(*ZOOM_RANGE)
    return make_affine(width, height, scale, tx, ty)


def transform_boxes(boxes: Boxes, matrix: np.ndarray, width: int, height: int) -> Optional[Boxes]:
    """Apply the same affine to all 5 boxes. None if any box is cut off too much."""
    sx, ox = matrix[0, 0], matrix[0, 2]
    sy, oy = matrix[1, 1], matrix[1, 2]
    out: Boxes = {}
    for site, (x1, y1, x2, y2) in boxes.items():
        raw = (sx * x1 + ox, sy * y1 + oy, sx * x2 + ox, sy * y2 + oy)
        clipped = clip_box(raw, width, height)
        if clipped is None:
            return None
        raw_area = (raw[2] - raw[0]) * (raw[3] - raw[1])
        clipped_area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
        if clipped_area / raw_area < MIN_RETAINED_BOX_FRACTION:
            return None
        out[site] = clipped
    return out


def choose_transform(rng: random.Random, allow_original: bool = True) -> str:
    kinds = [k for k in TRANSFORM_WEIGHTS if allow_original or k != "original"]
    return rng.choices(kinds, weights=[TRANSFORM_WEIGHTS[k] for k in kinds], k=1)[0]


def pick_view(image: np.ndarray, boxes: Boxes, rng: random.Random, allow_original: bool = True):
    """One random view. Retries if a box is cut off, then falls back to the original."""
    height, width = image.shape[:2]
    for _ in range(9):
        kind = choose_transform(rng, allow_original)
        if kind == "original":
            return kind, image.copy(), boxes
        matrix = sample_transform(kind, width, height, rng)
        new_boxes = transform_boxes(boxes, matrix, width, height)
        if new_boxes is not None:
            warped = cv2.warpAffine(
                image, matrix, (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,  # black border suits MRI
            )
            return kind, warped, new_boxes
    return "original_fallback", image.copy(), boxes


def resize_sample(image: np.ndarray, boxes: Boxes):
    """Resize to OUT_SIZE x OUT_SIZE and scale all boxes to match."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    sx, sy = OUT_SIZE / w, OUT_SIZE / h
    return small, {s: (x1 * sx, y1 * sy, x2 * sx, y2 * sy) for s, (x1, y1, x2, y2) in boxes.items()}


# ------------------------------- OUTPUT -------------------------------
def write_voc_xml(
    output_path: Path, image_path: Path, width: int, height: int,
    boxes: Boxes, states: dict[str, str], source_xml: Path, transform_name: str,
):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "Axial_5ROI"
    ET.SubElement(root, "filename").text = image_path.name
    ET.SubElement(root, "path").text = str(image_path)
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "AISSLab_Axial_5roi"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "1"
    ET.SubElement(root, "segmented").text = "0"
    ET.SubElement(root, "source_xml").text = str(source_xml)
    ET.SubElement(root, "transform").text = transform_name

    for site in SITES:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = f"{site}_{states[site]}"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        for tag, value in zip(COORDS, boxes[site]):
            ET.SubElement(bndbox, tag).text = f"{value:.2f}"   # floats: boxes are only ~2 px wide

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def draw_preview(image: np.ndarray, boxes: Boxes, out_path: Path):
    big = cv2.resize(image, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE,
                     interpolation=cv2.INTER_NEAREST)
    preview = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    for site in SITES:
        x1, y1, x2, y2 = [int(round(v * PREVIEW_SCALE)) for v in boxes[site]]
        cv2.rectangle(preview, (x1, y1), (x2, y2), PREVIEW_COLORS[site], 1)
    cv2.imwrite(str(out_path), preview)


# ------------------------------ DISCOVERY -----------------------------
def discover_pairs():
    pairs, missing_xml = [], []
    for image_path in sorted(SOURCE_ROOT.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        xml_path = image_path.with_suffix(".xml")
        if not xml_path.exists():  # try a case-insensitive stem match
            matches = [
                p for p in image_path.parent.glob("*.xml")
                if p.stem.lower() == image_path.stem.lower()
            ]
            xml_path = matches[0] if matches else xml_path
        if xml_path.exists():
            pairs.append((image_path, xml_path))
        else:
            missing_xml.append(image_path)
    return pairs, missing_xml


def assign_splits(patients, rng: random.Random):
    """Assign each PATIENT to train/val/test."""
    patients = sorted(patients)  # sort first so the shuffle is reproducible
    rng.shuffle(patients)
    n_train = int(len(patients) * TRAIN_RATIO)
    n_val = int(len(patients) * VAL_RATIO)
    return {
        p: "train" if i < n_train else "val" if i < n_train + n_val else "test"
        for i, p in enumerate(patients)
    }


# -------------------------------- MAIN --------------------------------
def main():
    rng = random.Random(SEED)
    np.random.seed(SEED)

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"Source folder not found: {SOURCE_ROOT}\n"
            "Edit SOURCE_ROOT at the top of this script."
        )

    if CLEAN_OUTPUT_FIRST and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    image_out = OUTPUT_ROOT / "images"
    xml_out = OUTPUT_ROOT / "annotations"
    preview_out = OUTPUT_ROOT / "previews"
    image_out.mkdir(parents=True, exist_ok=True)
    xml_out.mkdir(parents=True, exist_ok=True)
    if SAVE_PREVIEWS:
        preview_out.mkdir(parents=True, exist_ok=True)

    pairs, missing_xml = discover_pairs()
    print(f"[INFO] Found image/XML pairs: {len(pairs)}")
    print(f"[INFO] Images without matching XML: {len(missing_xml)}")

    # ---- Load and validate every source image ----
    records = []  # (key, image_path, xml_path, image, boxes, states)
    skipped = Counter()
    for image_path, xml_path in pairs:
        ann, reason = parse_annotation(xml_path)
        if ann is None:
            skipped[reason] += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[WARN] Cannot read image: {image_path}")
            skipped["unreadable_image"] += 1
            continue
        h, w = image.shape[:2]
        boxes = ann.boxes
        if (ann.width, ann.height) != (w, h):
            print(f"[WARN] Size mismatch {image_path}: XML {ann.width}x{ann.height}, "
                  f"image {w}x{h}; scaling boxes to the image size.")
            sx, sy = w / ann.width, h / ann.height
            boxes = {s: (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy) for s, b in boxes.items()}
        key = image_path.relative_to(SOURCE_ROOT).as_posix()
        records.append((key, image_path, xml_path, image, boxes, ann.states))

    # ---- Split by patient (top-level folder 0001..0500) ----
    patient_split = assign_splits({r[0].split("/")[0] for r in records}, rng)

    manifest_rows = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for key, image_path, xml_path, image, base_boxes, states in records:
        patient = key.split("/")[0]
        base_split = patient_split[patient]
        source_id = Path(key).with_suffix("").as_posix().replace("/", "__")

        # train: several random views. val/test: the original + one shifted view.
        if base_split == "train":
            views = [("train", *pick_view(image, base_boxes, rng)) for _ in range(TRAIN_COPIES)]
        else:
            views = [
                (base_split, "original", image.copy(), base_boxes),
                (f"{base_split}_shift",
                 *pick_view(image, base_boxes, rng, allow_original=False)),
            ]

        for v, (split, kind, out_image, out_boxes) in enumerate(views):
            out_image, out_boxes = resize_sample(out_image, out_boxes)

            file_stem = f"{source_id}__{kind}__v{v}"
            out_image_path = image_out / f"{file_stem}.png"
            out_xml_path = xml_out / f"{file_stem}.xml"

            if not cv2.imwrite(str(out_image_path), out_image):
                print(f"[WARN] Failed to write image: {out_image_path}")
                continue

            write_voc_xml(out_xml_path, out_image_path, OUT_SIZE, OUT_SIZE,
                          out_boxes, states, xml_path, kind)
            if SAVE_PREVIEWS:
                draw_preview(out_image, out_boxes, preview_out / f"{file_stem}.png")

            row = {
                "image": str(out_image_path),
                "annotation": str(out_xml_path),
                "source_image": str(image_path),
                "source_xml": str(xml_path),
                "source_key": key,
                "patient": patient,
                "split": split,
                "transform": kind,
                "width": OUT_SIZE,
                "height": OUT_SIZE,
            }
            for site in SITES:
                for coord, value in zip(COORDS, out_boxes[site]):
                    row[f"{site}_{coord}"] = round(value, 3)
                row[f"{site}_state"] = states[site]
            manifest_rows.append(row)
            counts[split][kind] += 1

    # ---- Sanity check: each patient lives in exactly one base split ----
    seen: dict[str, set[str]] = {}
    for row in manifest_rows:
        seen.setdefault(row["patient"], set()).add(row["split"].removesuffix("_shift"))
    assert all(len(s) == 1 for s in seen.values()), "patient leakage across splits!"

    # ---- Write CSVs ----
    box_fields = [f"{s}_{c}" for s in SITES for c in COORDS]
    state_fields = [f"{s}_state" for s in SITES]
    manifest_fields = (["image", "annotation", "source_image", "source_xml", "source_key",
                        "patient", "split", "transform", "width", "height"]
                       + box_fields + state_fields)
    split_fields = ["image", "annotation", "source_key", "patient", "split", "transform"]
    for filename, fields in (("manifest.csv", manifest_fields), ("splits.csv", split_fields)):
        with (OUTPUT_ROOT / filename).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: row[k] for k in fields} for row in manifest_rows)

    # ---- Summary ----
    print("\n========== DATASET SUMMARY ==========")
    print(f"Valid source images:      {len(records)}")
    print(f"Output images written:    {len(manifest_rows)}")
    print(f"Missing XML:              {len(missing_xml)}")
    print(f"Rejected pairs:           {sum(skipped.values())}"
          + (f"  ({', '.join(f'{k} {n}' for k, n in sorted(skipped.items()))})" if skipped else ""))
    print(f"Output folder:            {OUTPUT_ROOT}")

    print("\nTransforms per split:")
    for split in SPLITS:
        if split in counts:
            detail = ", ".join(f"{k} {n}" for k, n in sorted(counts[split].items()))
            print(f"  {split:10s}: {detail}")

    print("\nSplit counts (images / patients):")
    for split in SPLITS:
        n_img = sum(r["split"] == split for r in manifest_rows)
        base = split.removesuffix("_shift")
        n_pat = sum(s == base for s in patient_split.values())
        print(f"  {split:10s}: {n_img:6d} images / {n_pat:4d} patients")

    print("\nReview previews/: the 5 colored boxes should sit on the 5 ROIs "
          "(CCS green, LLS orange, RLS blue, LFS yellow, RFS magenta).")
    print("Remember to set CLEAN_OUTPUT_FIRST = False if you do not want to wipe the output next run.")


if __name__ == "__main__":
    main()