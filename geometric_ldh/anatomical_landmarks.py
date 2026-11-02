
"""
Anatomical landmark estimation from axial lumbar MRI.

Classes:
    0 = BG
    1 = PE
    2 = CC
    3 = IVD

Algorithm:
    A, B: robust local peaks on the superior contour of PE,
          one on each side of the CC-derived midline.
    C: posterior intersection of the perpendicular ray from
       midpoint(A, B) with the PE mask.

IMPORTANT:
    This is a research candidate detector, not a clinically
    validated anatomical landmark detector.
    Coordinates are IMAGE coordinates, not patient laterality.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt


Point = Tuple[float, float]


@dataclass
class LandmarkResult:
    right_facet: Optional[Point]
    left_facet: Optional[Point]
    spinous_tip: Optional[Point]
    confidence: Dict[str, float]
    valid: bool
    reason: str


class AnatomicalLandmarkDetector:

    def __init__(
        self,
        pe_class=1,
        cc_class=2,
        min_component_area=20,
        close_kernel=3,
        smooth_kernel=11,
        peak_support_radius=5,
        peak_prominence_px=1.5,
        central_exclusion_ratio=0.08,
        ray_step=0.5,
        min_ray_hits=3,
    ):
        self.pe_class = pe_class
        self.cc_class = cc_class
        self.min_component_area = min_component_area
        self.close_kernel = close_kernel
        self.smooth_kernel = smooth_kernel
        self.peak_support_radius = peak_support_radius
        self.peak_prominence_px = peak_prominence_px
        self.central_exclusion_ratio = central_exclusion_ratio
        self.ray_step = ray_step
        self.min_ray_hits = min_ray_hits

    # --------------------------------------------------
    # Mask preprocessing
    # --------------------------------------------------

    def _clean_mask(self, binary):

        binary = binary.astype(np.uint8)

        k = max(1, self.close_kernel)
        kernel = np.ones((k, k), np.uint8)

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, 8
        )

        output = np.zeros_like(binary)

        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]

            if area >= self.min_component_area:
                output[labels == i] = 1

        return output

    @staticmethod
    def _centroid(mask):

        y, x = np.where(mask > 0)

        if len(x) == 0:
            return None

        return float(x.mean()), float(y.mean())

    # --------------------------------------------------
    # Build superior contour of PE
    # --------------------------------------------------

    def _build_superior_contour(self, pe):

        h, w = pe.shape

        contour = np.full(w, np.nan, dtype=np.float32)

        for x in range(w):

            ys = np.where(pe[:, x] > 0)[0]

            if len(ys) > 0:
                # Uppermost PE boundary at this x.
                contour[x] = float(ys.min())

        valid = np.isfinite(contour)

        if valid.sum() < 5:
            return None

        xs = np.arange(w)

        # Interpolate missing columns.
        contour = np.interp(
            xs,
            xs[valid],
            contour[valid]
        ).astype(np.float32)

        # Smooth the contour to suppress isolated high pixels.
        k = max(3, self.smooth_kernel)

        if k % 2 == 0:
            k += 1

        smooth = cv2.GaussianBlur(
            contour.reshape(1, -1),
            (k, 1),
            0
        ).reshape(-1)

        return smooth

    # --------------------------------------------------
    # Find robust peak on one side
    # --------------------------------------------------

    def _find_side_peak(
        self,
        pe,
        contour,
        mid_x,
        side
    ):

        h, w = pe.shape

        pe_x = np.where(pe > 0)[1]

        if len(pe_x) == 0:
            return None, 0.0

        x_min = int(pe_x.min())
        x_max = int(pe_x.max())

        pe_width = max(1, x_max - x_min + 1)

        exclusion = max(
            2,
            int(self.central_exclusion_ratio * pe_width)
        )

        if side == "left":
            start = x_min + self.peak_support_radius
            end = int(mid_x - exclusion)

        else:
            start = int(mid_x + exclusion)
            end = x_max - self.peak_support_radius

        start = max(0, start)
        end = min(w - 1, end)

        if end <= start:
            return None, 0.0

        candidates = []

        radius = self.peak_support_radius

        for x in range(start + radius, end - radius + 1):

            local = contour[x-radius:x+radius+1]

            if len(local) < 3:
                continue

            center_y = float(contour[x])

            left_y = float(contour[x-radius])
            right_y = float(contour[x+radius])

            # Positive prominence means the center is superior
            # (smaller image-y) than its neighborhood.
            prominence = (
                (left_y + right_y) / 2.0
                - center_y
            )

            # Require a local minimum in the smoothed contour.
            if center_y > float(np.min(local)) + 0.5:
                continue

            if prominence < self.peak_prominence_px:
                continue

            # Verify that the candidate is supported by actual PE pixels.
            local_pe = pe[
                :,
                max(0, x-radius):min(w, x+radius+1)
            ]

            support = int(np.count_nonzero(local_pe))

            if support < 3:
                continue

            # Score uses prominence and spatial support.
            # It does not simply select the smallest y coordinate.
            support_score = min(1.0, support / 25.0)

            score = (
                0.65 * min(1.0, prominence / 5.0)
                + 0.35 * support_score
            )

            candidates.append((score, x, center_y))

        if not candidates:
            return None, 0.0

        # Select the best-supported local peak.
        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        score, peak_x, peak_y = candidates[0]

        # Refine location using nearby PE contour samples.
        x0 = max(0, peak_x - radius)
        x1 = min(w, peak_x + radius + 1)

        local_x = np.arange(x0, x1)
        local_y = contour[x0:x1]

        threshold = peak_y + 1.5

        selected = local_y <= threshold

        if selected.sum() > 0:
            refined_x = float(local_x[selected].mean())
            refined_y = float(local_y[selected].mean())
        else:
            refined_x = float(peak_x)
            refined_y = float(peak_y)

        return (refined_x, refined_y), float(score)

 
    # --------------------------------------------------
    # Find C along perpendicular ray from AB
    # --------------------------------------------------

    def _find_spinous_point(self, pe, A, B):
        """
        Find C as the first sustained intersection of the
        perpendicular ray from midpoint AB with the posterior
        central PE structure.
        """

        ax, ay = A
        bx, by = B

        # Midpoint of AB.
        mid_x = (ax + bx) / 2.0
        mid_y = (ay + by) / 2.0

        dx = bx - ax
        dy = by - ay

        length = np.hypot(dx, dy)

        if length < 3:
            return None, 0.0

        # Unit normal perpendicular to AB.
        nx = -dy / length
        ny = dx / length

        # Point toward posterior (increasing image-y).
        if ny < 0:
            nx = -nx
            ny = -ny

        h, w = pe.shape
        max_dist = np.hypot(h, w)

        samples = []
        t = 0.0

        while t <= max_dist:

            x = mid_x + t * nx
            y = mid_y + t * ny

            xi = int(round(x))
            yi = int(round(y))

            if not (0 <= xi < w and 0 <= yi < h):
                break

            samples.append({
                "t": t,
                "x": x,
                "y": y,
                "inside_pe": bool(pe[yi, xi] > 0)
            })

            t += self.ray_step

        if not samples:
            return None, 0.0

        # Find the first continuous PE intersection.
        required_hits = max(3, self.min_ray_hits)

        run = []

        for sample in samples:

            if sample["inside_pe"]:
                run.append(sample)

                if len(run) >= required_hits:

                    # First point of the sustained intersection.
                    contact = run[0]

                    point = (
                        float(contact["x"]),
                        float(contact["y"])
                    )

                    confidence = min(
                        1.0,
                        len(run) / 10.0
                    )

                    return point, confidence

            else:
                run = []

        return None, 0.0


    # --------------------------------------------------
    # Main detector
    # --------------------------------------------------

    def detect(self, segmentation_mask):

        mask = np.asarray(segmentation_mask)

        if mask.ndim != 2:
            raise ValueError(
                "Expected a 2D segmentation class mask."
            )

        pe = self._clean_mask(
            mask == self.pe_class
        )

        cc = self._clean_mask(
            mask == self.cc_class
        )

        cc_center = self._centroid(cc)

        if cc_center is None:
            return LandmarkResult(
                None, None, None, {},
                False,
                "CC mask is empty."
            )

        if np.count_nonzero(pe) == 0:
            return LandmarkResult(
                None, None, None, {},
                False,
                "PE mask is empty."
            )

        mid_x = cc_center[0]

        contour = self._build_superior_contour(pe)

        if contour is None:
            return LandmarkResult(
                None, None, None, {},
                False,
                "Could not construct PE superior contour."
            )

        # A and B: one robust superior peak on each side.
        A, conf_A = self._find_side_peak(
            pe, contour, mid_x, "left"
        )

        B, conf_B = self._find_side_peak(
            pe, contour, mid_x, "right"
        )

        if A is None or B is None:

            return LandmarkResult(
                A,
                B,
                None,
                {
                    "A": round(conf_A, 3),
                    "B": round(conf_B, 3)
                },
                False,
                "Could not identify two supported PE peaks."
            )

        # Confirm A and B are on opposite sides of CC midline.
        if not (A[0] < mid_x < B[0]):

            return LandmarkResult(
                A, B, None,
                {
                    "A": round(conf_A, 3),
                    "B": round(conf_B, 3)
                },
                False,
                "A and B do not straddle the CC midline."
            )

        # C: perpendicular intersection with posterior PE.
        C, conf_C = self._find_spinous_point(
            pe, A, B
        )

        if C is None:

            return LandmarkResult(
                A, B, None,
                {
                    "A": round(conf_A, 3),
                    "B": round(conf_B, 3),
                    "C": 0.0
                },
                False,
                "Perpendicular ray did not find enough PE support."
            )

        # Basic geometric plausibility check.
        if C[1] <= max(A[1], B[1]):

            return LandmarkResult(
                A, B, C,
                {
                    "A": round(conf_A, 3),
                    "B": round(conf_B, 3),
                    "C": round(conf_C, 3)
                },
                False,
                "C is not posterior to A and B."
            )

        confidence = {
            "A": round(conf_A, 3),
            "B": round(conf_B, 3),
            "C": round(conf_C, 3)
        }

        # valid means geometric checks passed only.
        # It does not mean anatomical correctness is confirmed.
        return LandmarkResult(
            right_facet=B,
            left_facet=A,
            spinous_tip=C,
            confidence=confidence,
            valid=True,
            reason="Geometric checks passed; anatomical review required."
        )


# ------------------------------------------------------
# Visualization
# ------------------------------------------------------

def visualize_landmarks(
    image,
    seg_mask,
    result,
    output_path=None
):

    image = np.asarray(image)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(image, cmap="gray")

    # Overlay PE and CC boundaries.
    pe = seg_mask == 1
    cc = seg_mask == 2

    ax.contour(
        pe.astype(float),
        levels=[0.5],
        colors=["red"],
        linewidths=1.2
    )

    ax.contour(
        cc.astype(float),
        levels=[0.5],
        colors=["lime"],
        linewidths=1.2
    )

    A = result.left_facet
    B = result.right_facet
    C = result.spinous_tip

    if A is not None and B is not None:

        ax.plot(
            [A[0], B[0]],
            [A[1], B[1]],
            color="yellow",
            linewidth=2,
            label="AB"
        )

        ax.scatter(
            [A[0], B[0]],
            [A[1], B[1]],
            c=["cyan", "yellow"],
            s=65,
            edgecolors="black",
            zorder=5
        )

        ax.text(
            A[0] + 3, A[1] - 5,
            "A",
            color="cyan",
            fontsize=12,
            weight="bold"
        )

        ax.text(
            B[0] + 3, B[1] - 5,
            "B",
            color="yellow",
            fontsize=12,
            weight="bold"
        )

        # Draw perpendicular line from midpoint AB.
        midpoint = (
            (A[0] + B[0]) / 2,
            (A[1] + B[1]) / 2
        )

        if C is not None:
            ax.plot(
                [midpoint[0], C[0]],
                [midpoint[1], C[1]],
                color="white",
                linestyle="--",
                linewidth=1.5
            )

    if C is not None:

        ax.scatter(
            [C[0]],
            [C[1]],
            c="white",
            s=65,
            edgecolors="black",
            zorder=5
        )

        ax.text(
            C[0] + 3,
            C[1] + 5,
            "C - Spinous",
            color="white",
            fontsize=10,
            weight="bold"
        )

    status = (
        "GEOMETRIC CHECK PASSED"
        if result.valid
        else "UNRELIABLE"
    )

    ax.set_title(
        f"Landmarks: {status}\n{result.reason}"
    )

    ax.axis("off")

    if output_path:
        from pathlib import Path

        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight"
        )

    plt.close(fig)
