"""Paired transforms for image + mask.

Every spatial op is applied to both with the same parameters.
Masks always use NEAREST interpolation so labels stay discrete.
"""

import random

import numpy as np
import torch
from PIL import Image


# RAW_VALUES = [0, 50, 100, 150, 200]   # raw mask pixel values
# NUM_CLASSES = len(RAW_VALUES)
# CLASS_NAMES = ["Background", "PE", "CC", "AAP", "IVD"]   # index = class id

RAW_VALUES = [0, 50, 100, 200]                 # bỏ AAP (150)
NUM_CLASSES = len(RAW_VALUES)
CLASS_NAMES = ["Background", "PE", "CC", "IVD"]

# lookup table: raw pixel value -> class index (0..4)
_LUT = np.zeros(256, dtype=np.int64)
for idx, raw in enumerate(RAW_VALUES):
    _LUT[raw] = idx


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class DualResize:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else tuple(size)  # (w, h)

    def __call__(self, image, mask):
        return (image.resize(self.size, Image.BILINEAR),
                mask.resize(self.size, Image.NEAREST))


class DualRandomHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            return (image.transpose(Image.FLIP_LEFT_RIGHT),
                    mask.transpose(Image.FLIP_LEFT_RIGHT))
        return image, mask


class DualRandomRotation:
    def __init__(self, degrees=10, p=0.5):
        self.degrees, self.p = degrees, p

    def __call__(self, image, mask):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            return (image.rotate(angle, Image.BILINEAR),
                    mask.rotate(angle, Image.NEAREST))
        return image, mask


class DualToTensor:
    """Image -> float [1,H,W], standardized per image. Mask -> long [H,W] with values 0..4."""

    def __call__(self, image, mask):
        x = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0)
        x = (x - x.mean()) / (x.std() + 1e-6)
        y = torch.from_numpy(_LUT[np.array(mask, dtype=np.uint8)])
        return x, y