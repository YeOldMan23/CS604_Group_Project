import torch

class U2NetCustomLoss(torch.nn.Module):
    def __init__(self):
        super(U2NetCustomLoss, self).__init__()
        self.name = "U2Net Model(s) Custom Loss"
        self.bce_loss = torch.nn.BCELoss(size_average=True)

    def forward(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        loss0 = self.bce_loss(d0, labels_v)
        loss1 = self.bce_loss(d1, labels_v)
        loss2 = self.bce_loss(d2, labels_v)
        loss3 = self.bce_loss(d3, labels_v)
        loss4 = self.bce_loss(d4, labels_v)
        loss5 = self.bce_loss(d5, labels_v)
        loss6 = self.bce_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        return loss0, loss