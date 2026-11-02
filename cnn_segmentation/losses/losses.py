import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """CrossEntropy + soft Dice. Dice handles the background-heavy class imbalance."""

    def __init__(self, num_classes=5, dice_weight=1.0, eps=1e-6):
        super().__init__()
        self.num_classes, self.dice_weight, self.eps = num_classes, dice_weight, eps

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target)

        prob = logits.softmax(dim=1)
        onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        inter = (prob * onehot).sum(dim=(0, 2, 3))
        total = prob.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
        dice = (2 * inter + self.eps) / (total + self.eps)

        return ce + self.dice_weight * (1 - dice.mean())