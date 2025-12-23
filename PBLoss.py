import torch
import torch.nn as nn
import torch.nn.functional as F


class PBLoss(nn.Module):
    """
    Foreground-Background Loss (Focal + Tversky hybrid)
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.5, lam=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam = lam
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        y = targets
        p = probs.clamp(1e-6, 1-1e-6)

        # Pixel-wise focal-tversky hybrid term
        focal_pos = - self.alpha * (1 - p) ** self.gamma * y * torch.log(p)
        focal_neg = - self.beta * (p ** self.gamma) * (1 - y) * torch.log(1 - p)
        focal_term = focal_pos + focal_neg

        # Global Tversky term
        TP = (p * y).sum()
        FP = (p * (1 - y)).sum()
        FN = ((1 - p) * y).sum()
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        tversky_term = 1 - tversky_index

        # Combine
        loss = focal_term.mean() + self.lam * tversky_term
        return loss
