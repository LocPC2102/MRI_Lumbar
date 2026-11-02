"""
Loss, box decoding and box overlap helpers for the 5-site ROI detector.

Prediction layout: raw [B, S * V, G, G]   (S = 5 sites, V = 5 + num_classes values per site)
    channel = site * V + k,   k: 0 tx, 1 ty, 2 tw, 3 th, 4 objectness   (raw logits)

Boxes are [xmin, ymin, xmax, ymax] normalized to [0, 1].
gt_boxes is [B, S, 4] in SITES order. Every site has exactly one box per image.

Decoding (cell units, relative to the cell's top-left corner):
    center offset = 2 * sigmoid(t) - 0.5      range (-0.5, 1.5)
    width, height = sigmoid(t)                fraction of the image

Positive cells (per site, per image): the cell holding the box center plus its two nearest
neighbours (one horizontal, one vertical). A neighbour can predict a center slightly outside
its own cell (hence the (-0.5, 1.5) range), so a center sitting on a cell border is no longer
lost when the model picks the neighbouring cell.
"""

import torch
import torch.nn.functional as F

from .. import config as cfg


# ----------------------------- box helpers ----------------------------
def box_overlap(a: torch.Tensor, b: torch.Tensor):
    """Pairwise (row i of a vs row i of b): intersection, area of a, area of b."""
    lt = torch.maximum(a[:, :2], b[:, :2])
    rb = torch.minimum(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(dim=1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(dim=1)
    return inter, area_a, area_b


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter, area_a, area_b = box_overlap(a, b)
    return inter / (area_a + area_b - inter + 1e-7)


def encode_targets(boxes: torch.Tensor, grid: int):
    """
    GT boxes [N, 4] -> 3 positive cells per box.
    Returns gi, gj [N, 3] (own cell, horizontal neighbour, vertical neighbour),
    valid [N, 3] (False when a neighbour falls outside the grid),
    center [N, 2] in cell units and size [N, 2] as a fraction of the image.
    """
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 * grid
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 * grid
    size = torch.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], dim=1)
    ci = cx.floor().long().clamp(0, grid - 1)
    cj = cy.floor().long().clamp(0, grid - 1)
    dx = (cx - ci >= 0.5).long() * 2 - 1             # nearest horizontal neighbour: -1 or +1
    dy = (cy - cj >= 0.5).long() * 2 - 1             # nearest vertical neighbour: -1 or +1
    gi = torch.stack([ci, ci + dx, ci], dim=1)
    gj = torch.stack([cj, cj, cj + dy], dim=1)
    valid = (gi >= 0) & (gi < grid) & (gj >= 0) & (gj < grid)
    return gi.clamp(0, grid - 1), gj.clamp(0, grid - 1), valid, torch.stack([cx, cy], dim=1), size


# ------------------------------- decoding -----------------------------
def split_sites(raw: torch.Tensor) -> torch.Tensor:
    """[B, S * V, G, G] -> [B * S, V, G, G]  (one row per image-site pair, image-major)."""
    grid = raw.shape[-1]
    return raw.reshape(-1, raw.shape[1] // cfg.NUM_SITES, grid, grid)


def center_offset(t: torch.Tensor) -> torch.Tensor:
    """Center logits -> offset from the cell's top-left corner, in cell units."""
    return 2.0 * torch.sigmoid(t) - 0.5


def decode_cells(sites: torch.Tensor, rows: torch.Tensor, gi: torch.Tensor, gj: torch.Tensor):
    """Decode the box stored at cell (gi, gj) of row `rows` of `sites` -> [P, 4] xyxy, normalized."""
    grid = sites.shape[-1]
    cell = sites[rows, :, gj, gi]                    # [P, V]
    cx = (gi + center_offset(cell[:, 0])) / grid
    cy = (gj + center_offset(cell[:, 1])) / grid
    w = torch.sigmoid(cell[:, 2])
    h = torch.sigmoid(cell[:, 3])
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def decode_best_boxes(raw: torch.Tensor):
    """
    Highest-objectness cell of every site.
    Returns boxes [B, S, 4], scores [B, S], gi [B, S], gj [B, S].
    """
    batch = raw.shape[0]
    sites = split_sites(raw)
    grid = sites.shape[-1]
    score, flat = sites[:, 4].flatten(1).max(dim=1)  # flat index = gj * grid + gi
    gj, gi = flat // grid, flat % grid
    rows = torch.arange(sites.shape[0], device=raw.device)
    boxes = decode_cells(sites, rows, gi, gj).clamp(0, 1)
    shape = (batch, cfg.NUM_SITES)
    return boxes.view(*shape, 4), torch.sigmoid(score).view(shape), gi.view(shape), gj.view(shape)


# -------------------------------- loss --------------------------------
def detection_loss(raw: torch.Tensor, gt_boxes: torch.Tensor):
    """Returns (total_loss tensor, dict of float components). Every site counts equally."""
    sites = split_sites(raw)                         # [N, V, G, G], N = B * S
    gt = gt_boxes.reshape(-1, 4)                     # [N, 4], same image-major order
    n, _, grid, _ = sites.shape
    gi, gj, valid, center, size = encode_targets(gt, grid)

    # Flatten the (up to 3) positive cells of every site into one list of P positives.
    rows = torch.arange(n, device=raw.device).unsqueeze(1).expand_as(gi)[valid]
    pi, pj = gi[valid], gj[valid]
    cell = sites[rows, :, pj, pi]                    # [P, V]

    # Box regression at every positive cell. Offset and size are both compared in cell
    # units so thin boxes are not drowned out.
    target_xy = center[rows] - torch.stack([pi, pj], dim=1)
    coord = (F.l1_loss(center_offset(cell[:, 0:2]), target_xy)
             + F.l1_loss(torch.sigmoid(cell[:, 2:4]) * grid, size[rows] * grid))
    iou_loss = (1.0 - box_iou(decode_cells(sites, rows, pi, pj), gt[rows])).mean()

    # Objectness: the positive cells of a site vs all its other cells, averaged separately.
    target = torch.zeros(n, grid, grid, device=raw.device)
    target[rows, pj, pi] = 1.0
    bce = F.binary_cross_entropy_with_logits(sites[:, 4], target, reduction="none")
    pos = bce[target > 0.5].mean()
    neg = (bce * (1.0 - target)).sum() / (1.0 - target).sum()

    total = (cfg.LAMBDA_OBJ * (pos + cfg.LAMBDA_NOOBJ * neg)
             + cfg.LAMBDA_COORD * coord
             + cfg.LAMBDA_IOU * iou_loss)

    parts = {
        "loss": total.item(),
        "obj_pos": pos.item(),
        "obj_neg": neg.item(),
        "coord": coord.item(),
        "iou_loss": iou_loss.item(),
    }
    return total, parts


# ------------------------------ smoke test ----------------------------
if __name__ == "__main__":
    grid, num_sites, batch = cfg.GRID_SIZE, cfg.NUM_SITES, 2
    values = 5 + cfg.NUM_CLASSES

    # Two images, 5 thin neighbouring boxes each.
    gt = torch.zeros(batch, num_sites, 4)
    for b in range(batch):
        for s in range(num_sites):
            cx, cy, w, h = 0.30 + 0.10 * s + 0.05 * b, 0.45 + 0.10 * b, 0.03, 0.12
            gt[b, s] = torch.tensor([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    # Build the "perfect" prediction by hand (every positive cell holds the exact box):
    # loss must be ~0 and decoding must return gt.
    perfect = torch.zeros(batch, num_sites * values, grid, grid)
    for b in range(batch):
        for s in range(num_sites):
            gi, gj, valid, center, size = encode_targets(gt[b, s:s + 1], grid)
            base = s * values
            perfect[b, base + 4] = -10.0
            for k in range(3):
                if not valid[0, k]:
                    continue
                i, j = gi[0, k], gj[0, k]
                offset = center[0] - torch.stack([i, j]).to(center.dtype)
                perfect[b, base + 4, j, i] = 10.0
                perfect[b, base:base + 2, j, i] = torch.logit(((offset + 0.5) / 2).clamp(1e-4, 1 - 1e-4))
                perfect[b, base + 2:base + 4, j, i] = torch.logit(size[0].clamp(1e-4, 1 - 1e-4))

    loss, _ = detection_loss(perfect, gt)
    boxes, _, _, _ = decode_best_boxes(perfect)
    print("perfect prediction  loss:", round(loss.item(), 5), "(expect ~0)")
    print("decoded == gt       :", torch.allclose(boxes, gt, atol=1e-3))

    # Random prediction: loss must be finite and gradients must flow.
    random_raw = torch.randn(batch, num_sites * values, grid, grid, requires_grad=True)
    loss, parts = detection_loss(random_raw, gt)
    loss.backward()
    print("random prediction   :", {k: round(v, 3) for k, v in parts.items()})
    print("gradient is finite  :", bool(torch.isfinite(random_raw.grad).all()))