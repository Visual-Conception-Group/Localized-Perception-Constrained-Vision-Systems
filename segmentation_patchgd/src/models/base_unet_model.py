import torch
import torch.nn as nn
from tqdm import tqdm 


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return {'out':outputs}


def train_base(
        model1     = None,
        optimizer1 = None,
        loss_fn    = None,
        loader     = None,
        device     = 'cuda',
        mode       = 'train' # OR 'eval'
    ):
    """
    to_pil_img = torchvision.transforms.functional.to_pil_image
    to_pil_img(x[0]).save('input.png')
    to_pil_img(y[0]).save('mask.png')
    exit()

    - read the image, 
    - break into patches, create batch of patches
    - get results from the model
    - create the mask 
    - calculate the loss
    """
    epoch_loss = 0.0
    size = len(loader)

    if mode=='train':
        model1.train()
    else:
        print("**** In Eval Mode")
        model1.eval()

    print("::Loader Size", size)
    
    for x, y in tqdm(loader):
        x = x.to(device, dtype=torch.float32) # can be batched image
        y = y.to(device, dtype=torch.float32)

        if mode=='train':
            optimizer1.zero_grad()
            y_pred = model1(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer1.step()
        else:
            with torch.no_grad():
                y_pred = model1(x)
                loss = loss_fn(y_pred, y)
        epoch_loss += loss.item()

    print(f"---- Last Batch Input Shape: {x.shape}, {y.shape} || Pred Shape: {y_pred.shape}")
    epoch_loss = epoch_loss/size
    return epoch_loss   


if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))
    f = build_unet()
    y = f(x)
    print(y.shape)


