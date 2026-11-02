"""Central configuration for the 5-ROI detection task."""

from pathlib import Path

import torch

# ------------------------------- paths --------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "dataset_5roi_256"

MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs_256"
CHECKPOINT_PATH = OUTPUT_DIR / "best.pt"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# ----------------------------- data / model ---------------------------
IMAGE_SIZE = 256                                # must match datasets/build_dataset.py
IN_CHANNELS = 1
SITES = ("CCS", "LLS", "RLS", "LFS", "RFS")       # slot order, same as datasets/dataset.py
NUM_SITES = len(SITES)
NUM_CLASSES = 0                                   # per-site grade logits (added later)
GRID_SIZE = IMAGE_SIZE // 16                # backbone has two 2x2 pools (stride 4) -> 16x16

# -------------------------------- loss --------------------------------
LAMBDA_OBJ = 1.0      # objectness (positive cell + negative cells)
LAMBDA_NOOBJ = 1.0    # weight of the negative-cell term inside objectness
LAMBDA_COORD = 5.0    # L1 on center offset and box size (both in cell units)
LAMBDA_IOU = 5.0      # (1 - IoU) on the decoded box

# ------------------------------ training ------------------------------
SEED = 42
EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0                 # 0 is safest on Windows
EARLY_STOP_PATIENCE = 20        # epochs without val_shift IoU improvement
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"