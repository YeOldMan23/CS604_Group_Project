import torch

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, beta = 0.3, gamma = 4/3, smooth = 1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.beta  = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds, target):
        preds = preds.view(-1)
        target = target.view(-1)

        true_pos = torch.sum(preds * target)
        false_neg = torch.sum(preds * (1-target))
        false_pos = torch.sum((1-preds)*target)

        # Calculate tversky loss first
        tversky = (true_pos + self.smooth)/(true_pos + self.beta *false_neg + (1-self.beta)*false_pos + self.smooth)
        tversky_loss = 1 - tversky

        focal_tversky_loss = torch.pow(1 - tversky_loss, 1 / self.gamma)
        
        return focal_tversky_loss
