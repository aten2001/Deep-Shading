import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import pytorch_ssim

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False

class GI_Model(torch.nn.Module):
    def __init__(self):
        super(GI_Model, self).__init__()
        self.act = nn.LeakyReLU(0.01, inplace = True)
        self.pool = nn.AvgPool2d(2, stride=2)
        self.ssim_loss = pytorch_ssim.SSIM(window_size = 8)

        self.down_0_conv = nn.Conv2d(in_channels=7,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.down_1_conv = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,groups=2)
        self.down_2_conv = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1,groups=4)
        self.down_3_conv = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,groups=8)
        self.down_4_conv = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,groups=16)
        
        self.up_4_to_3 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False,groups=256)
        self.up_3_conv = nn.Conv2d(in_channels=384,out_channels=128,kernel_size=3,stride=1,padding=1,groups=8)

        self.up_3_to_2 = nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False,groups=128)
        self.up_2_conv = nn.Conv2d(in_channels=192,out_channels=64,kernel_size=3,stride=1,padding=1,groups=4)

        self.up_2_to_1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False,groups=64)
        self.up_1_conv = nn.Conv2d(in_channels=96,out_channels=32,kernel_size=3,stride=1,padding=1,groups=2)

        self.up_1_to_0 = nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1,bias=False,groups=32)
        self.up_0_conv = nn.Conv2d(in_channels=48,out_channels=1,kernel_size=3,stride=1,padding=1)

        nn.init.normal_(self.down_0_conv.weight, std=0.01)
        nn.init.normal_(self.down_1_conv.weight, std=0.01)
        nn.init.normal_(self.down_2_conv.weight, std=0.01)
        nn.init.normal_(self.down_3_conv.weight, std=0.01)
        nn.init.normal_(self.down_4_conv.weight, std=0.01)

        nn.init.normal_(self.up_3_conv.weight, std=0.01)
        nn.init.normal_(self.up_2_conv.weight, std=0.01)
        nn.init.normal_(self.up_1_conv.weight, std=0.01)
        nn.init.normal_(self.up_0_conv.weight, std=0.01)

        nn.init.normal_(self.down_0_conv.bias, std=0.01)
        nn.init.normal_(self.down_1_conv.bias, std=0.01)
        nn.init.normal_(self.down_2_conv.bias, std=0.01)
        nn.init.normal_(self.down_3_conv.bias, std=0.01)
        nn.init.normal_(self.down_4_conv.bias, std=0.01)

        nn.init.normal_(self.up_3_conv.bias, std=0.01)
        nn.init.normal_(self.up_2_conv.bias, std=0.01)
        nn.init.normal_(self.up_1_conv.bias, std=0.01)
        nn.init.normal_(self.up_0_conv.bias, std=0.01)

        nn.init.constant_(self.up_4_to_3.weight, 0.25)
        nn.init.constant_(self.up_3_to_2.weight, 0.25)
        nn.init.constant_(self.up_2_to_1.weight, 0.25)
        nn.init.constant_(self.up_1_to_0.weight, 0.25)

        freeze_layer(self.up_4_to_3)
        freeze_layer(self.up_3_to_2)
        freeze_layer(self.up_2_to_1)
        freeze_layer(self.up_1_to_0)

    def forward(self, image):
        
        image = self.down_0_conv(image)
        image = self.act(image)
        image_0 = image.clone()
        image = self.pool(image)
        
        image = self.down_1_conv(image)
        image = self.act(image)
        image_1 = image.clone()
        image = self.pool(image)
        
        image = self.down_2_conv(image)
        image = self.act(image)
        image_2 = image.clone()
        image = self.pool(image)
        
        image = self.down_3_conv(image)
        image = self.act(image)
        image_3 = image.clone()
        image = self.pool(image)
        
        image = self.down_4_conv(image)
        image = self.act(image)
        
        image = self.up_4_to_3(image)
        image = torch.cat((image, image_3), 1)
        image = self.up_3_conv(image)
        image = self.act(image)
        
        image = self.up_3_to_2(image)
        image = torch.cat((image, image_2), 1)
        image = self.up_2_conv(image)
        image = self.act(image)
        
        image = self.up_2_to_1(image)
        image = torch.cat((image, image_1), 1)
        image = self.up_1_conv(image)
        image = self.act(image)
        
        image = self.up_1_to_0(image)
        image = torch.cat((image, image_0), 1)
        image = self.up_0_conv(image)
        image = self.act(image)

        return image
        
    def loss(self, output_image, ground_truth):
        loss = self.ssim_loss(output_image, ground_truth)
        return (1.0 - loss) / 2.0