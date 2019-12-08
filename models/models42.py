import torch
import torch.nn as nn

class DoubleCNN(nn.Module):
    def __init__(self, ch_in=1, ch_out=1):
        super().__init__()
        self.dCNN=nn.Sequential(nn.Conv2d(ch_in,ch_out,kernel_size=3,padding=1),nn.BatchNorm2d(ch_out),nn.ReLU(inplace=True),nn.Conv2d(ch_out,ch_out,kernel_size=3,padding=1),nn.BatchNorm2d(ch_out),nn.ReLU(inplace=True))
    def forward(self,out):
        return self.dCNN(out)

class UNet_LR(nn.Module):
    def __init__(self):
        super(UNet_LR,self).__init__()
        self.c1=DoubleCNN(1,64)
        self.c2=DoubleCNN(64,128)
        self.c3=DoubleCNN(128,256)
        self.c4=DoubleCNN(256,512)
        self.cm1=DoubleCNN(512,1024)
        self.cm2=nn.Sequential(nn.Conv2d(1024,1024,kernel_size=1,padding=2),nn.BatchNorm2d(1024),nn.ReLU(inplace=True))
        self.cm3=DoubleCNN(1024,512)
        self.c5=DoubleCNN(512,256)
        self.c6=DoubleCNN(256,128)
        self.c7=DoubleCNN(128,64)
        self.c8=DoubleCNN(64,1)
        self.DownSampling=nn.MaxPool2d(2,2)
        self.UpSampling=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
    def forward(self,out):
        out=self.c1(out)               # 1x64
        out=self.DownSampling(out)     # Image size W/2 H/2
        out=self.c2(out)               # 64x128
        out=self.DownSampling(out)     # Image size W/4 H/4
        out=self.c3(out)               # 128x256
        out=self.DownSampling(out)     # Image size W/8 H/8
        out=self.c4(out)               # 256x512
        out=self.DownSampling(out)     # Image size W/16 H/16
        out=self.cm1(out)              # 512x1024
        out=self.cm2(out)              # 1024x1024
        out=self.UpSampling(out)       # Image size W/8 H/8
        
        out=self.cm3(out)              # 1024x512
        out=self.c5(out)               # 512x256
        
        out=self.UpSampling(out)       # Image size W/4 H/4
        out=self.c6(out)               # 256x128
        out=self.UpSampling(out)       # Image size W/2 H/2
        out=self.c7(out)               # 128x64
        out=self.UpSampling(out)       # Image size W H
        out=self.c8(out)               # 64x1
        return out
