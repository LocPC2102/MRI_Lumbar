"""Audit AISSLab Axial Segmentation: pair images with masks and write a manifest CSV.

Run from the project root:
    python src/cnn_segmentation/datasets/audit_dataset.py
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

FIELDS = ["patient_id", "image_path", "mask_path", "width", "height",
          "mask_values", "pair_valid", "error"]


def audit(data_dir: Path, out_csv: Path) -> None:
    rows = []
    sizes, value_sets, per_patient = Counter(), Counter(), Counter()

    for pdir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        files = list(pdir.glob("*.png"))
        masks = {f.stem[:-2]: f for f in files if f.stem.endswith("_M")}
        images = {f.stem: f for f in files if not f.stem.endswith("_M")}

        for stem in sorted(images.keys() | masks.keys()):
            img_p, msk_p = images.get(stem), masks.get(stem)
            row = dict(patient_id=pdir.name,
                       image_path=img_p.as_posix() if img_p else "",
                       mask_path=msk_p.as_posix() if msk_p else "",
                       width="", height="", mask_values="", pair_valid=False, error="")

            if img_p is None or msk_p is None:
                row["error"] = "missing image" if img_p is None else "missing mask"
            else:
                try:
                    with Image.open(img_p) as img, Image.open(msk_p) as msk:
                        if img.size != msk.size:
                            row["error"] = f"size mismatch {img.size} vs {msk.size}"
                        else:
                            vals = tuple(int(v) for v in np.unique(np.array(msk)))
                            row.update(width=img.size[0], height=img.size[1],
                                       mask_values=",".join(map(str, vals)), pair_valid=True)
                            sizes[img.size] += 1
                            value_sets[vals] += 1
                            per_patient[pdir.name] += 1
                except Exception as e:
                    row["error"] = f"unreadable: {e}"
            rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    valid = sum(r["pair_valid"] for r in rows)
    print(f"Patients      : {len({r['patient_id'] for r in rows})}")
    print(f"Valid pairs   : {valid} / {len(rows)}")
    print(f"Sizes         : {dict(sizes)}")
    print(f"Mask values   : {dict(value_sets)}")
    print(f"Pairs/patient : {dict(Counter(per_patient.values()))}")
    print(f"Invalid rows  : {[(r['image_path'] or r['mask_path'], r['error']) for r in rows if not r['pair_valid']]}")
    print(f"Manifest saved: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="DATA/raw/AISSLab_Axial/Segmentation")
    ap.add_argument("--out", default="DATA/processed/segmentation_manifest.csv")
    args = ap.parse_args()
    audit(Path(args.data_dir), Path(args.out))