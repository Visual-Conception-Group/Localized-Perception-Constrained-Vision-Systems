
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        intersection_cardinality = (inputs * targets).sum()
        union_cardinality = inputs.sum() + targets.sum()
        dice = ((2.*intersection_cardinality) + smooth)/(union_cardinality + smooth)
        loss = 1 - dice
        # print(f"Int: {intersection_cardinality.detach().cpu().numpy()}, Union: {union_cardinality.detach().cpu().numpy()}, Loss: {loss.detach().cpu().numpy()}")
        return loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.loss_dice = DiceLoss()

    def forward(self, inputs, targets, smooth=1, use_sigmoid:bool=True):

        # print(inputs.max())
        #comment out if your model contains a sigmoid or equivalent activation layer
        if use_sigmoid:
            inputs = torch.sigmoid(inputs)
        # print(inputs.max())
        # print(inputs.shape)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        dice_loss = self.loss_dice(inputs, targets)
        # print("Dice loss:", dice_loss)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        return Dice_BCE

class DiceOnly(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceOnly, self).__init__()
        self.loss_dice = DiceLoss()

    def forward(self, inputs, targets, smooth=1, use_sigmoid:bool=True):
        # print(inputs.max())
        #comment out if your model contains a sigmoid or equivalent activation layer
        if use_sigmoid:
            inputs = torch.sigmoid(inputs)
        # print(inputs.max())
        # print(inputs.shape)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        dice_loss = self.loss_dice(inputs, targets)
        return dice_loss
    
class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, use_sigmoid:bool=True):
        if use_sigmoid:
            inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        loss = BCE
        return loss
    
if __name__=="__main__":
    loss = DiceBCELoss()

    # a = torch.ones(10)
    # b = torch.ones(10)

    a = torch.tensor([1,1,1,1,1,0,0,0,0,0]).float()
    b = torch.tensor([0.7,0.7,0.7,0.7,0.7,0,0,0,0,0]).float() # all correct - but ratio (0)
    # b = torch.tensor([1,1,1,1,1,0,0,0,0,0]).float() # all correct (0)
    # b = torch.tensor([0,0,0,1,1,0,0,0,0,0]).float() # less area covered, but correct (30) (1.37)
    # b = torch.tensor([0,0,0,0,0,0,0,0,0,0]).float() # no area covered, no wrong, no correct (50) (1.86)
    # b = torch.tensor([0,0,0,1,1,1,1,1,1,1]).float() # more area covered, but wrong (80) (1.17)
    # b = torch.tensor([0,0,0,0,0,1,1,1,1,1]).float() # all area covered, all incorrect (100) (1.5)

    print(a)
    print(b)

    val = loss(a, b, use_sigmoid=False)
    print("loss value:", val)