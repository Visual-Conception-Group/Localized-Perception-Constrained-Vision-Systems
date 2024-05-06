import torch.nn as nn
import torch

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class z_block_v1(nn.Module):
    """
    64 => 1
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.outputs = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.outputs(x)
        return x
    
class z_block_v2(nn.Module):
    """
    64 => 64 => 1
    """
    def __init__(self, in_c, out_c, interim_c = 64, retain_dim=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, interim_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(interim_c)

        self.output = nn.Conv2d(interim_c, out_c, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        
        self.outSize = 512
        self.retain_dim = retain_dim

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.output(x)
        return x


class z_block_v3(nn.Module):
    """
    64 => 32 => 1
    """
    def __init__(self, in_c, out_c, interim_c = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, interim_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(interim_c)

        self.relu = nn.ReLU()
        self.output = nn.Conv2d(interim_c, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.output(x)
        return x

class z_block_v4(nn.Module):
    """
    64 => 32 => 16 => 1
    """
    def __init__(self, in_c, out_c, interim_c = 32, interim_c2 = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, interim_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(interim_c)
       
        self.conv2 = nn.Conv2d(interim_c, interim_c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(interim_c2)
       
        self.relu = nn.ReLU()
        self.output = nn.Conv2d(interim_c2, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.output(x)
        return x
    
class z_block_v5(nn.Module):
    """
    64 => 32 => 16 => 1

    dynamically changing theta2 network
    """
    def __init__(self, in_c, out_c):
        super().__init__()

        # max to not reduce it further than number of classes
        interim_c = max(int(in_c/2), out_c)
        interim_c2 = max(int(interim_c/2), out_c)

        self.conv1 = nn.Conv2d(in_c, interim_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(interim_c)
       
        self.conv2 = nn.Conv2d(interim_c, interim_c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(interim_c2)
       
        self.relu = nn.ReLU()
        self.output = nn.Conv2d(interim_c2, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.output(x)
        return x

# TEST PGD MODULE
if __name__=="__main__":
    model = z_block_v4(64, 1)
    print(model)