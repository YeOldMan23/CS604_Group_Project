import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        preds: [batch_size, num_classes, H, W]
        targets: [batch_size, H, W] with values in {0, 1, ..., num_classes - 1}
        """

        # Compute Dice coefficient per class per sample
        dims = (0, 2, 3)  # dimensions to sum over: batch, height, width

        intersection = torch.sum(preds * targets, dims)
        union = preds.sum(dim=dims) + targets.sum(dim=dims)
        

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()  # [num_classes]

        # Average over classes
        return dice_loss