import torch.nn as nn
import constants
from torchvision.models import resnet18, resnet50, mobilenet_v2
import mobileNetV2
import torch.nn as nn
import constants
import torch
from torchvision.models import resnet18, resnet50
import numpy as np
    
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
def getModel1():
    # model = mobilenet_v2()
    model = resnet50()
    # model.classifier = Identity()
    model.fc = Identity()
    return model

def getModel2():
    model = resnet50(num_classes=constants.numClasses)
    model.conv1 = nn.Conv2d(4, 64, 2)
    return model
