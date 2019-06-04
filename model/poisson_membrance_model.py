import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import pdb
import torch

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [  nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(out_size),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class Generator(nn.Module):
    def __init__(self,ngf=32):
        # G用来从U生成Alpha
        super(Generator, self).__init__()
        self.down1 = UNetDown(1, ngf, normalize=False)
        self.down2 = UNetDown(ngf, ngf*2)
        self.down3 = UNetDown(ngf*2, ngf*4)
        self.down4 = UNetDown(ngf*4, ngf*4, dropout=0.5)
        self.down5 = UNetDown(ngf*4, ngf*4, dropout=0.5)
        self.down6 = UNetDown(ngf*4, ngf*4, normalize=False, dropout=0.5)

        self.up1 = UNetUp(ngf*4, ngf*4, dropout=0.5)
        self.up2 = UNetUp(ngf*8, ngf*4, dropout=0.5)
        self.up3 = UNetUp(ngf*8, ngf*4, dropout=0.5)
        self.up4 = UNetUp(ngf*8, ngf*2, dropout=0.5)
        self.up5 = UNetUp(ngf*4, ngf, dropout=0.5)
        self.up6 = nn.Upsample(scale_factor=2)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(ngf*2, 1, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, Input):
        #Unet 格式的模型
        d1 = self.down1(Input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6,d5)
        u2 = self.up2(u1,d4)
        u3 = self.up3(u2,d3)
        u4 = self.up4(u3,d2)
        u5 = self.up5(u4,d1)

        return self.final(u5)

class Discriminator(nn.Module):
    def __init__(self, in_channels=2,ndf = 32):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, ndf, normalization=False),
            *discriminator_block(ndf, ndf*2),
            *discriminator_block(ndf*2, ndf*4),
            *discriminator_block(ndf*4, ndf*8),
            *discriminator_block(ndf*8, ndf*16),
            nn.Conv2d(ndf*16, 1, 4, 1, 0, bias=False)
        )

    def forward(self, Input):
        # Concatenate image and condition image by channels to produce input
        return self.model(Input)




