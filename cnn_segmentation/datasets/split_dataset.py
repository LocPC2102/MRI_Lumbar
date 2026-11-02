"""Split the segmentation manifest into train/val/test by patient.

All slices of one patient go to the same split, so there is no patient leakage.

Run from the project root:
    python src/cnn_segmentation/datasets/split_dataset.py
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def split(manifest: Path, out_dir: Path, ratios=(0.7, 0.15, 0.15), seed=42) -> None:
    with open(manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if r["pair_valid"] == "True"]

    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)

    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)

    n = len(patients)
    n_train, n_val = int(ratios[0] * n), int(ratios[1] * n)
    groups = {
        "train": patients[:n_train],
        "val": patients[n_train:n_train + n_val],
        "test": patients[n_train + n_val:],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, pids in groups.items():
        split_rows = [r for p in pids for r in by_patient[p]]
        with open(out_dir / f"seg_{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(split_rows)
        total += len(split_rows)
        print(f"{name:5s}: {len(pids):3d} patients, {len(split_rows):4d} pairs")

    # Checks
    sets = [set(p) for p in groups.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "patient overlap!"
    assert total == len(rows), "pair count mismatch!"
    print(f"OK: no overlap, {total} pairs in total (seed={seed})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="DATA/processed/segmentation_manifest.csv")
    ap.add_argument("--out-dir", default="DATA/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    split(Path(args.manifest), Path(args.out_dir), seed=args.seed)