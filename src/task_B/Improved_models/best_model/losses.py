"""Loss functions used by the improved Task B model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss with optional per-class alpha weights."""

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss from logits and class targets."""
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = alpha_t * focal_weight
        loss = focal_weight * ce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


class SupConLoss(nn.Module):
    """Supervised contrastive loss on normalized embeddings."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute supervised contrastive loss."""
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)

        sim = torch.matmul(features, features.T) / self.temperature
        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float()

        logits_mask = 1.0 - torch.eye(batch_size, device=features.device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_count = mask.sum(dim=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / (pos_count + 1e-8)
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        return -mean_log_prob[valid].mean()
