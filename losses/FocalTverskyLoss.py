import torch
import sys

# Add the sys to the s

class FocalTverskyLoss(torch.nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=True,
        alpha=0.6,
        beta=0.4,
        smooth=1,
        gamma=4 / 3,
        class_weights=None,
    ):
        super(FocalTverskyLoss, self).__init__()
        self.name = "Focal Tversky Loss"
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gamma = gamma
        self.class_weights = class_weights

        # Keep BG weight at 1
        if self.class_weights is not None:
            self.no_class = len(class_weights)

    def forward(self, inputs, targets):
        # Ok for multiclass multilabel since the entire image becomes the same dimensions
        class_input = inputs.view(-1)
        class_target = targets.view(-1)

        # True Positive, False Negative
        TP = (class_input * class_target).sum()
        FP = ((1 - class_target) * class_input).sum()
        FN = (class_target * (1 - class_input)).sum()

        Tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        FocalTversky = (1 - Tversky) ** (1 / self.gamma)

        return FocalTversky