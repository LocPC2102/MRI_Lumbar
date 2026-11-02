from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

Point = Tuple[float, float]
BG, PE, CC, IVD = 0, 1, 2, 3


# ============================================================
# 1. Segmentation cleaning
# ============================================================

def _components(binary):
    return cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)[:3]


def _largest(binary):
    n, labels, stats = _components(binary)
    if n <= 1:
        return np.zeros(binary.shape, np.uint8)
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def _fill_holes(binary):
    cnts, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(binary.shape, np.uint8)
    cv2.drawContours(out, cnts, -1, 1, thickness=-1)
    return out


def clean_segmentation(mask, pe_max_dist=20.0, pe_min_frac=0.05):
    """
    IVD, CC : largest connected component only.
    PE      : largest component + components that are big enough (>= pe_min_frac of the
              largest) AND within pe_max_dist px of the CC; far-away blobs are dropped.
    Then fill holes and close 1-px gaps (only into background pixels).
    """
    mask = np.asarray(mask)
    ivd, cc = _largest(mask == IVD), _largest(mask == CC)

    n, labels, stats = _components(mask == PE)
    pe = np.zeros(mask.shape, np.uint8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest, max_area = 1 + int(np.argmax(areas)), float(areas.max())
        dist = cv2.distanceTransform((1 - cc).astype(np.uint8), cv2.DIST_L2, 3) if cc.any() else None
        for i in range(1, n):
            comp = labels == i
            keep = i == largest
            if not keep and dist is not None:
                keep = (stats[i, cv2.CC_STAT_AREA] >= pe_min_frac * max_area
                        and float(dist[comp].min()) <= pe_max_dist)
            if keep:
                pe[comp] = 1

    out = np.zeros(mask.shape, np.uint8)
    for cls, part in ((IVD, ivd), (CC, cc), (PE, pe)):
        out[(_fill_holes(part) == 1) & (out == 0)] = cls
    kernel = np.ones((3, 3), np.uint8)
    for cls in (IVD, CC, PE):
        closed = cv2.morphologyEx((out == cls).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        out[(closed == 1) & (out == 0)] = cls
    return out


# ============================================================
# 2. Result
# ============================================================

@dataclass
class LdhResult:
    image_name: str
    grade_num: int
    zone_code: str
    zone_name: str
    lateralization: str
    ab_px: float
    ab_mm: float
    gap_mm: float                 # baseline -> disc distance when the disc does not cross it
    cd_px: float
    cd_mm: float
    index_pct: float              # ab / cd * 100
    apex: Point                   # deepest IVD point (b)
    A: Point
    B: Point
    C: Point
    boxes: Dict[str, List[float]]
    herniated: Optional[np.ndarray] = None   # (N, 2) [x, y] IVD pixels past the baseline

    @property
    def grade(self):
        return f"Grade {self.grade_num}"


# ============================================================
# 3. Engine
# ============================================================

class LandmarkLdhEngine:
    def __init__(
        self,
        pixel_spacing_mm: float = 0.6,
        cross_tol_px: float = 1.0,
        min_cross_px: int = 15,
        lateral_margin: float = 0.5,     # IVD search range past A/B, as a fraction of |AB|
        grade1_mm: float = 3.0,
    ):
        self.mm = pixel_spacing_mm
        self.tol = cross_tol_px
        self.min_cross = min_cross_px
        self.margin = lateral_margin
        self.grade1_mm = grade1_mm

    @staticmethod
    def _frame(A: Point, B: Point):
        """Origin, |AB|, unit vector along AB, unit normal pointing posterior."""
        a = np.array(A, dtype=float)
        vec = np.array(B, dtype=float) - a
        L = float(np.hypot(*vec))
        u = vec / max(L, 1e-6)
        n = np.array([-u[1], u[0]])
        if n[1] < 0:
            n = -n
        return a, L, u, n

    @staticmethod
    def _zone(apex_x, i1_x, i2_x, boxes):
        """Zone from the herniation x position vs the centres of RFS, RLS, CCS, LLS, LFS."""
        c = {s: (boxes[s][0] + boxes[s][2]) / 2.0 if s in boxes else 128.0
             for s in ("RFS", "RLS", "CCS", "LLS", "LFS")}
        r_fs, r_ls, c_cs, l_ls, l_fs = sorted(c.values())

        m1, m2, m3, m4 = (r_fs + r_ls) / 2, (r_ls + c_cs) / 2, (c_cs + l_ls) / 2, (l_ls + l_fs) / 2
        efz_r = r_fs - 0.6 * max(r_ls - r_fs, 8.0)
        efz_l = l_fs + 0.6 * max(l_fs - l_ls, 8.0)
        x = 0.70 * apex_x + 0.15 * i1_x + 0.15 * i2_x

        if x < efz_r:
            return "R_EFZ", "Extraforaminal Zone", "Right"
        if x < m1:
            return "RFS", "Foraminal Zone", "Right"
        if x < m2:
            return "RLS", "Subarticular Zone", "Right"
        if x <= m3:
            return "CCS", "Central Canal Zone", "Central"
        if x <= m4:
            return "LLS", "Subarticular Zone", "Left"
        if x <= efz_l:
            return "LFS", "Foraminal Zone", "Left"
        return "L_EFZ", "Extraforaminal Zone", "Left"

    def analyze(self, seg_mask, boxes, A: Point, B: Point, C: Point,
                image_name="axial_mri") -> Optional[LdhResult]:
        """seg_mask must already be cleaned. Returns None if the geometry is not valid."""
        if A[0] > B[0]:
            A, B = B, A
        a, L, u, n = self._frame(A, B)
        if L < 3:
            return None

        cd_px = float((np.array(C, dtype=float) - a) @ n)
        if cd_px <= 1:                                   # C must be posterior to the baseline
            return None

        ivd = (seg_mask == IVD).astype(np.uint8)
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(ivd, 8)
        if n_lab <= 1:
            return None
        ys, xs = np.where(labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))

        t = (xs - a[0]) * u[0] + (ys - a[1]) * u[1]      # position along AB
        d = (xs - a[0]) * n[0] + (ys - a[1]) * n[1]      # depth past AB (+ = posterior)
        roi = (t >= -self.margin * L) & (t <= (1 + self.margin) * L)
        if not roi.any():
            return None
        xs, ys, t, d = xs[roi], ys[roi], t[roi], d[roi]

        crossing = d > self.tol
        herniated = None
        if crossing.sum() >= self.min_cross:
            herniated = np.stack([xs[crossing], ys[crossing]], axis=1)
            i = int(np.argmax(np.where(crossing, d, -np.inf)))
            ab_px = float(d[i])
            index = ab_px / cd_px
            if index > 1.0:                              # deeper than the canal -> landmarks wrong
                return None
            gap_mm = 0.0
            grade = 2 if index <= 0.3 else 3

            base = crossing & (d <= self.tol + 1.5)      # where the herniation meets the baseline
            if base.any():
                bt = np.where(base, t, np.nan)
                i1, i2 = int(np.nanargmin(bt)), int(np.nanargmax(bt))
            else:
                i1 = i2 = i
            i1_x, i2_x = float(xs[i1]), float(xs[i2])
        else:
            i = int(np.argmax(d))
            ab_px, index = 0.0, 0.0
            gap_mm = float(-d[i]) * self.mm
            grade = 1 if gap_mm <= self.grade1_mm else 0
            i1_x = i2_x = float(xs[i])

        apex = (float(xs[i]), float(ys[i]))
        if grade == 0:
            zone = ("CCS", "Central Canal Zone", "Central")
        else:
            zone = self._zone(apex[0], i1_x, i2_x, boxes)

        return LdhResult(
            image_name=image_name, grade_num=grade,
            zone_code=zone[0], zone_name=zone[1], lateralization=zone[2],
            ab_px=ab_px, ab_mm=ab_px * self.mm, gap_mm=gap_mm,
            cd_px=cd_px, cd_mm=cd_px * self.mm, index_pct=index * 100.0,
            apex=apex, A=A, B=B, C=C, boxes=boxes, herniated=herniated,
        )

    # --------------------------------------------------------
    def render(self, image_gray, seg_mask, r: LdhResult, output_path):
        """Clean image (thin lines, small markers, no text) + side panel with numbers and legend."""
        h, w = image_gray.shape[:2]
        A, B, C = r.A, r.B, r.C
        a, L, u, n = self._frame(A, B)

        BGC, PANEL, TXT, DIM = "#12121e", "#1b1b2f", "#e8e8f5", "#9aa0c8"
        PE_C, CC_C, IVD_C, HERN_C, AB_C = "#ff4dff", "#ffd600", "#00e5ff", "#ff1744", "#ff9100"
        box_color = {"CCS": "#2ecc71", "LLS": "#3498db", "RLS": "#3498db",
                     "LFS": "#e67e22", "RFS": "#e67e22"}

        fig = plt.figure(figsize=(11.5, 6.4), dpi=150, facecolor=BGC)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.62], wspace=0.03,
                              left=0.01, right=0.99, top=0.94, bottom=0.02)
        ax, pn = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

        # ---- image ----
        ax.imshow(image_gray, cmap="gray", vmin=0, vmax=255)
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        overlay[seg_mask == PE] = [0.85, 0.10, 0.85, 0.16]
        overlay[seg_mask == CC] = [1.00, 0.85, 0.00, 0.20]
        overlay[seg_mask == IVD] = [0.00, 0.80, 1.00, 0.16]
        hmask = None
        if r.herniated is not None:
            hmask = np.zeros((h, w), dtype=np.uint8)
            hmask[r.herniated[:, 1], r.herniated[:, 0]] = 1
            overlay[hmask > 0] = [1.0, 0.09, 0.27, 0.70]
        ax.imshow(overlay)

        for cls, color in ((PE, PE_C), (CC, CC_C), (IVD, IVD_C)):
            m = seg_mask == cls
            if m.any():
                ax.contour(m.astype(float), levels=[0.5], colors=[color], linewidths=1.0)
        if hmask is not None:
            ax.contour(hmask.astype(float), levels=[0.5], colors=["white"], linewidths=1.2)

        for site, color in box_color.items():
            if site in r.boxes:
                x0, y0, x1, y1 = r.boxes[site]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor=color, linestyle="--", linewidth=0.8, alpha=0.85))

        ax.plot([A[0] - 40 * u[0], B[0] + 40 * u[0]], [A[1] - 40 * u[1], B[1] + 40 * u[1]],
                color="yellow", linestyle=":", linewidth=0.9, alpha=0.6, zorder=4)
        ax.plot([A[0], B[0]], [A[1], B[1]], color="yellow", linewidth=1.3, zorder=5)

        foot_c = a + float((np.array(C) - a) @ u) * u                    # cd: c -> C
        ax.plot([foot_c[0], C[0]], [foot_c[1], C[1]], color="white", linestyle="--",
                linewidth=1.0, zorder=6)

        apex = np.array(r.apex)                                          # ab / gap: a -> b
        foot_a = a + float((apex - a) @ u) * u
        ax.plot([foot_a[0], apex[0]], [foot_a[1], apex[1]], color=AB_C, linewidth=1.6,
                linestyle="-" if r.ab_px > 0 else "--", zorder=8)

        for p, c in ((A, "cyan"), (B, "yellow"), (C, "white")):
            ax.scatter([p[0]], [p[1]], c=c, s=16, edgecolors="black", linewidths=0.6, zorder=9)
        ax.scatter([foot_a[0]], [foot_a[1]], c=AB_C, s=8, edgecolors="black", linewidths=0.4, zorder=9)
        ax.scatter([apex[0]], [apex[1]], marker="*", c="gold", s=45, edgecolors="black",
                   linewidths=0.5, zorder=10)
        ax.set_title(r.image_name, color=TXT, fontsize=9, pad=4)
        ax.axis("off")

        # ---- side panel: grade, measurements, legend ----
        pn.set_xlim(0, 1)
        pn.set_ylim(0, 1)
        pn.axis("off")
        pn.add_patch(Rectangle((0, 0), 1, 1, facecolor=PANEL, edgecolor="#3a3a5a", linewidth=1))

        st = {"y": 0.955}
        STEP = 0.0285

        def text(x, s, color=TXT, size=7.5, bold=False):
            pn.text(x, st["y"], s, color=color, fontsize=size, family="monospace", va="center",
                    fontweight="bold" if bold else "normal")

        def header(s):
            st["y"] -= STEP * 0.4
            text(0.04, s, color=DIM, size=7, bold=True)
            st["y"] -= STEP

        def row(kind, color, label):
            y = st["y"]
            if kind in ("line", "dash"):
                pn.plot([0.04, 0.13], [y, y], color=color, linewidth=1.6,
                        linestyle="-" if kind == "line" else "--")
            elif kind == "dot":
                pn.scatter([0.085], [y], c=color, s=16, edgecolors="black", linewidths=0.6)
            elif kind == "star":
                pn.scatter([0.085], [y], marker="*", c=color, s=45, edgecolors="black", linewidths=0.5)
            elif kind == "fill":
                pn.add_patch(Rectangle((0.04, y - 0.009), 0.09, 0.018, facecolor=color,
                                       edgecolor="white", linewidth=0.5))
            elif kind == "box":
                pn.add_patch(Rectangle((0.04, y - 0.009), 0.09, 0.018, fill=False, edgecolor=color,
                                       linestyle="--", linewidth=1.0))
            text(0.17, label)
            st["y"] -= STEP

        badge = {0: "#27ae60", 1: "#f39c12", 2: "#e67e22", 3: "#c0392b"}[r.grade_num]
        pn.add_patch(Rectangle((0.03, st["y"] - 0.026), 0.94, 0.052, facecolor=badge,
                               edgecolor="white", linewidth=0.8))
        text(0.05, f"{r.grade.upper()}  |  {r.zone_name} ({r.lateralization})",
             color="white", size=7.8, bold=True)
        st["y"] -= STEP * 1.6

        header("MEASUREMENTS")
        for label, value in (("Site ", r.zone_code), ("ab   ", f"{r.ab_mm:.1f} mm"),
                             ("gap  ", f"{r.gap_mm:.1f} mm"), ("cd   ", f"{r.cd_mm:.1f} mm"),
                             ("ab/cd", f"{r.index_pct / 100:.2f}")):
            text(0.05, f"{label} : {value}")
            st["y"] -= STEP

        header("LEGEND")
        row("line", PE_C, "PE outline")
        row("line", CC_C, "CC outline")
        row("line", IVD_C, "IVD outline")
        row("fill", HERN_C, "IVD part past line A-B")
        row("line", "yellow", "Baseline A-B")
        row("dot", "cyan", "A  articular process")
        row("dot", "yellow", "B  articular process")
        row("dot", "white", "C  spinous process")
        row("line", AB_C, "ab: line A-B to apex")
        row("dash", AB_C, "gap: disc not crossing A-B")
        row("dash", "white", "cd: line A-B to C")
        row("star", "gold", "apex (deepest disc point)")
        row("box", box_color["CCS"], "ROI  CCS")
        row("box", box_color["LLS"], "ROI  LLS / RLS")
        row("box", box_color["LFS"], "ROI  LFS / RFS")

        plt.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close(fig)