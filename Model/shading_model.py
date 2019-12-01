import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import pytorch_ssim



class Shading_Model(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.act = nn.LeakyReLU(0.01, )
        self.pool = nn.AvgPool2d(2, stride=2)
        self.ssim_loss = pytorch_ssim.SSIM(window_size = 8)

        self.down_0_conv = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding=1)
        self.down_1_conv = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,groups=2)
        self.down_2_conv = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,groups=4)
        self.down_3_conv = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1,groups=8)
        self.down_4_conv = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,groups=16)
        self.down_5_conv = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,groups=32)
        
        self.up_5_to_4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=4,stride=2,padding=1,groups=256)
        self.up_4_conv = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1,groups=16)
        self.up_4_to_3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=4,stride=2,padding=1,groups=128)
        self.up_3_conv = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1,groups=8)
        self.up_3_to_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=4,stride=2,padding=1,groups=64)
        self.up_2_conv = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,groups=4)
        self.up_2_to_1 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=4,stride=2,padding=1,groups=32)
        self.up_1_conv = nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=1,groups=2)
        self.up_1_to_0 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=4,stride=2,padding=1,groups=16)
        self.up_0_conv = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=3,stride=1,padding=1)

    def forward(self, input_normal, input_position , input_ground_truth):
        
        image = torch.cat((input_normal, input_position), 1)
        
        if torch.cuda.is_available():
            image = imag.cuda()

        image = self.down_0_conv(image)
        image = self.act(image)
        image_0 = image
        image = self.poll(image)
        
        image = self.down_1_conv(image)
        image = self.act(image)
        image_1 = image
        image = self.poll(image)
        
        image = self.down_2_conv(image)
        image = self.act(image)
        image_2 = image
        image = self.poll(image)
        
        image = self.down_3_conv(image)
        image = self.act(image)
        image_3 = image
        image = self.poll(image)
        
        image = self.down_4_conv(image)
        image = self.act(image)
        image_4 = image
        image = self.poll(image)
        
        image = self.down_5_conv(image)
        image = self.act(image)
        
        image = self.up_5_to_4(image)
        image = torch.cat((image, image_4), 1)
        image = self.up_4_conv(image)
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
        
        if torch.cuda.is_available():
            input_ground_truth = input_ground_truth.cuda()
        
#         loss = self.ssim_loss(image,input_ground_truth)