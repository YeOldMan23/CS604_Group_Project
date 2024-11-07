import torch
import torch.nn.functional as F
import numpy as np

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=5, alpha=None, reduction='mean', ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha  # Can be None, scalar, or Tensor
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        device = logits.device

        # Convert targets to one-hot encoding
        targets = targets.long()

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1).clamp(min=1e-7, max=1 - 1e-7)  # [batch_size, num_classes, H, W]

        # Gather probabilities of the true class
        pt = (probs * targets).sum(dim=1)  # [batch_size, H, W]

        # Compute the focal loss components
        log_pt = torch.log(pt)
        focal_term = (1 - pt) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets].to(device)  # [batch_size, H, W]
            else:
                alpha_t = self.alpha
            loss = -alpha_t * focal_term * log_pt
        else:
            loss = -focal_term * log_pt

        # Handle ignore_index
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
            loss = loss * valid_mask.float()

            if self.reduction == 'mean':
                loss = loss.sum() / valid_mask.float().sum()
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        return loss
