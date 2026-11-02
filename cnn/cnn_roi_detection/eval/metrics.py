"""
Per-site localization metrics.
Boxes are [N, S, 4] normalized xyxy (S = number of sites, SITES order).
Pixel errors are measured on the IMAGE_SIZE x IMAGE_SIZE image.
"""

import torch

from .. import config as cfg
from ..loss.losses import box_iou, box_overlap


def per_item_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    cell_hit: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Metric of every (image, site) pair. Each returned tensor is [N, S].
    cell_hit [N, S]: the predicted cell is one of the site's positive cells.
    """
    n, s, _ = pred.shape
    p, g = pred.reshape(-1, 4), gt.reshape(-1, 4)
    inter, _, gt_area = box_overlap(p, g)

    def center(box):
        return (box[:, :2] + box[:, 2:]) / 2

    def size(box):
        return box[:, 2:] - box[:, :2]

    return {
        "iou": box_iou(p, g).view(n, s),
        "coverage": (inter / gt_area.clamp(min=1e-7)).view(n, s),   # share of GT inside the prediction
        "center_err_px": ((center(p) - center(g)) * cfg.IMAGE_SIZE).norm(dim=1).view(n, s),
        "size_err_px": ((size(p) - size(g)).abs() * cfg.IMAGE_SIZE).mean(dim=1).view(n, s),
        "cell_hit": cell_hit.float(),
    }


def summarize(values: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Average over images. Keys: "ALL" (mean over sites) and each site name."""
    groups = {"ALL": slice(None), **{site: i for i, site in enumerate(cfg.SITES)}}
    summary = {}
    for name, sel in groups.items():
        iou = values["iou"][:, sel]
        summary[name] = {
            "mean_iou": iou.mean().item(),
            "iou@0.5": (iou >= 0.5).float().mean().item(),
            "iou@0.7": (iou >= 0.7).float().mean().item(),
            "coverage": values["coverage"][:, sel].mean().item(),
            "center_err_px": values["center_err_px"][:, sel].mean().item(),
            "size_err_px": values["size_err_px"][:, sel].mean().item(),
            "cell_acc": values["cell_hit"][:, sel].mean().item(),
        }
    return summary


def constant_box_iou(mean_box: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """
    Mean IoU of always predicting the train-mean box of each site
    (the "guess the usual place" baseline). mean_box [S, 4], gt [N, S, 4].
    """
    n = gt.shape[0]
    expanded = mean_box.unsqueeze(0).expand(n, -1, -1)
    iou = box_iou(expanded.reshape(-1, 4), gt.reshape(-1, 4)).view(n, -1)
    return {"ALL": iou.mean().item(), **{s: iou[:, i].mean().item() for i, s in enumerate(cfg.SITES)}}