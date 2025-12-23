import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePBLoss(nn.Module):
    """
    Adaptive Foreground-Background Loss (P-B Loss).
    - Adapts alpha, beta, pos_weight based on foreground ratio r.
    - Uses pixel-level focal terms + global Tversky regularizer.
    """
    def __init__(self,
                 r_min=0.01, r_max=0.5,
                 beta_extra=0.4,   # s in mapping
                 gamma=1.5,
                 lam=0.5,
                 smooth=1e-6,
                 pos_weight_cap=50.0,
                 per_image=True,
                 running_avg_momentum=0.98,
                 use_running_global=False):
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.beta_extra = beta_extra
        self.gamma = gamma
        self.lam = lam
        self.smooth = smooth
        self.pos_weight_cap = pos_weight_cap
        self.per_image = per_image
        self.use_running_global = use_running_global

        # running global estimate of r (for stability / if the dataset is noisy)
        self.register_buffer('running_r', torch.tensor(0.1))
        self.momentum = running_avg_momentum

    def _map_beta_alpha(self, r):
        # r in [r_min, r_max], clamp first
        r_clamped = torch.clamp(r, self.r_min, self.r_max)
        # normalized in [0,1]
        t = (r_clamped - self.r_min) / (self.r_max - self.r_min + 1e-12)
        # invert t because small r -> bigger beta
        inv = 1.0 - t
        beta = 0.5 + self.beta_extra * inv
        alpha = 1.0 - beta
        return alpha, beta

    def forward(self, logits, targets):
        """
        logits: Tensor (B,1,H,W) or (B,H,W)
        targets: same shape, float {0,1}
        """
        # standardize shapes
        if logits.dim() == 4 and logits.size(1) == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.sigmoid(logits.unsqueeze(1)) if logits.dim() == 3 else torch.sigmoid(logits)

        y = targets
        if y.dim() == 3:
            y = y.unsqueeze(1)

        # compute per-image foreground ratio
        per_image_r = y.view(y.size(0), -1).mean(dim=1)  # in [0,1]

        # optionally update running global r
        batch_mean_r = per_image_r.mean().detach()
        if self.use_running_global:
            self.running_r = self.running_r * self.momentum + batch_mean_r * (1 - self.momentum)

        # which r to use
        if self.per_image:
            # r_for_map = per_image_r.view(-1, 1, 1, 1)  # shape (B,1,1,1)
            alpha, beta = self._map_beta_alpha(per_image_r)  # returns (B,) tensors
            alpha = alpha.view(-1, 1, 1, 1)
            beta = beta.view(-1, 1, 1, 1)
        else:
            r_global = self.running_r if self.use_running_global else batch_mean_r
            alpha_scalar, beta_scalar = self._map_beta_alpha(r_global)
            alpha = alpha_scalar
            beta = beta_scalar

        p = probs.clamp(1e-6, 1-1e-6)

        # Pixel-wise focal-like terms (with alpha/beta weight)
        # pos term (foreground)
        focal_pos = - (alpha * ((1 - p) ** self.gamma) * y * torch.log(p))
        # neg term (background)
        focal_neg = - (beta * (p ** self.gamma) * (1 - y) * torch.log(1 - p))
        focal_term = focal_pos + focal_neg
        focal_loss = focal_term.mean()

        # global Tversky term
        if self.per_image:
            # compute per image sums
            p_flat = p.view(p.size(0), -1)
            y_flat = y.view(y.size(0), -1)
            TP = (p_flat * y_flat).sum(dim=1)
            FP = (p_flat * (1 - y_flat)).sum(dim=1)
            FN = ((1 - p_flat) * y_flat).sum(dim=1)
            # alpha/beta currently per-image or scalar
            if isinstance(alpha, torch.Tensor):
                alpha_flat = alpha.view(-1)
                beta_flat = beta.view(-1)
            else:
                alpha_flat = alpha
                beta_flat = beta
            TI = (TP + self.smooth) / (TP + alpha_flat * FP + beta_flat * FN + self.smooth)
            tversky_term = (1 - TI).mean()
        else:
            # global aggregation
            TP = (p * y).sum()
            FP = (p * (1 - y)).sum()
            FN = ((1 - p) * y).sum()
            TI = (TP + self.smooth) / (TP + alpha * FP + beta * FN + self.smooth)
            tversky_term = 1 - TI

        # combine
        loss = focal_loss + self.lam * tversky_term

        # compute recommended pos_weight (for optional BCEWithLogits usage elsewhere)
        # pos_weight = clamp((1 - r) / (r + eps))
        # r_for_pw = per_image_r if self.per_image else (self.running_r if self.use_running_global else batch_mean_r)
        # pos_weight = ((1.0 - r_for_pw) / (r_for_pw + 1e-6)).clamp(max=self.pos_weight_cap)
        # return scalar loss, and optionally some stats for logging
        return loss
