import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class CNN_4L(nn.Module):
    def __init__(self):
        super(CNN_4L,self).__init__()
        ### Patch Extraction n1=64 c=1 f1=9
        self.block1=nn.Conv2d(1,64,kernel_size=9,padding=2)
        ### Non-Linear Mapping n2=n1=64 f2=1
        self.block2=nn.Conv2d(64,64,kernel_size=1,padding=2)
        ### Image Reconstruction n3=n2=32 f3=5
        self.block3=nn.Conv2d(64,32,kernel_size=5,padding=2)
        ### Image Reconstruction n4=n=32 f3=3
        self.block4=nn.Conv2d(32,1,kernel_size=3,padding=1)
        self.RELU=nn.ReLU()
    def forward(self,out):
        out=self.block1(out)
        out=self.RELU(out)
        out=self.block2(out)
        out=self.RELU(out)
        out=self.block3(out)
        out=self.RELU(out)
        out=self.block4(out)
        return out
    
class CNN_3L(nn.Module):
    def __init__(self):
        super(CNN_3L,self).__init__()
        ### Patch Extraction n1=64 c=1 f1=9
        self.block1=nn.Conv2d(1,64,kernel_size=9,padding=2)
        ### Non-Linear Mapping n2=n1=64 f2=1
        self.block2=nn.Conv2d(64,32,kernel_size=1,padding=2)
        ### Image Reconstruction n3=n2=32 f3=5
        self.block3=nn.Conv2d(32,1,kernel_size=5,padding=2)
        self.RELU=nn.ReLU()
    def forward(self,out):
        out=self.block1(out)
        out=self.RELU(out)
        out=self.block2(out)
        out=self.RELU(out)
        out=self.block3(out)
        return out
    

class CNN_XL(nn.Module):
    def __init__(self):
        super(CNN_XL,self).__init__()
        self.block1=nn.Conv2d(1,16,kernel_size=5,padding=2)
        self.block2=nn.Conv2d(16,32,kernel_size=3,padding=2)
        self.block3=nn.Conv2d(32,64,kernel_size=3,padding=2)
        self.block4=nn.Conv2d(64,128,kernel_size=3,padding=2)
        self.block5=nn.Conv2d(128,64,kernel_size=3,padding=2)
        self.block6=nn.Conv2d(64,32,kernel_size=3,padding=2)
        self.block7=nn.Conv2d(32,16,kernel_size=3,padding=2)
        self.block8=nn.Conv2d(16,1,kernel_size=1,padding=2)
        self.RELU=nn.ReLU()
    def forward(self,out):
        out=self.block1(out)
        out=self.RELU(out)
        out=self.block2(out)
        out=self.RELU(out)
        out=self.block3(out)
        out=self.RELU(out)
        out=self.block4(out)
        out=self.RELU(out)
        out=self.block5(out)
        out=self.RELU(out)
        out=self.block6(out)
        out=self.RELU(out)
        out=self.block7(out)
        out=self.RELU(out)
        out=self.block8(out)
        return out
