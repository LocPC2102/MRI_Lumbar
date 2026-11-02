import torch


@torch.no_grad()
def evaluate(model, loader, device, num_classes=5):
    """Return per-class IoU and Dice (NaN for a class that never appears in target or prediction)."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, device=device)   # rows: target, cols: prediction
    for x, y in loader:
        pred = model(x.to(device)).argmax(dim=1).flatten()
        idx = y.to(device).flatten() * num_classes + pred
        cm += torch.bincount(idx, minlength=num_classes ** 2).view(num_classes, num_classes)

    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    iou = tp / (tp + fp + fn)
    dice = 2 * tp / (2 * tp + fp + fn)
    return iou.cpu(), dice.cpu()