from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch 
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34
import torch.optim as optim

### BaseNet is a nn.Module and represents a simple CNN
class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.apply(self.initialize_weights)

        self.network = nn.Sequential(
            
        nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1), # [C, 12, 130, 130]
        nn.MaxPool2d(kernel_size=2),  # [C, 12, 65, 65]
        nn.ReLU(), # [C, 12, 65, 65]
        nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1), # [C, 24, 67, 67]
        nn.MaxPool2d(kernel_size=2), # [C, 24, 32, 32]
        nn.ReLU(),
    
        nn.Flatten(), # [C, 24576]
        nn.Linear(in_features=32 * 32 * 24, out_features=84) # [C, 84]
        )

    def initialize_weights(self): # custom weight initialization wasnt used but might be handy
        if isinstance(self, nn.Conv2d): # Kaiming for convolutions wieghts, constant for biases
            nn.init.kaiming_uniform_(self.weight.data,nonlinearity='relu')
            if self.bias is not None:
                nn.init.constant_(self.bias.data, 0)
        elif isinstance(self, nn.BatchNorm2d): # constant for BatchNorm
            nn.init.constant_(self.weight.data, 1)
            nn.init.constant_(self.bias.data, 0)
        elif isinstance(self, nn.Linear): # Kaiming for FCL wieghts, constant for biases
            nn.init.kaiming_uniform_(self.weight.data)
            nn.init.constant_(self.bias.data, 0)

    def forward(self, xb): # the network output is the log_softmax of the [C, 84] 
        return torch.log_softmax(self.network(xb), dim=1)


### ResNet18 is a nn.Module
class ResNet18(nn.Module): 
    def __init__(self):
        super().__init__()
        self.network = resnet18(pretrained=False, num_classes=84) # it is built by Pytorch 
        # The first layer is replaced to work for black and white images (1 input channel instead of 3)
        self.network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False) 

    def forward(self, xb):
        return torch.log_softmax(self.network(xb), dim=1) # the network output is the log_softmax of the the probabilities


### ResNet34 is a nn.Module and represents a simple CNN
class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = resnet34(pretrained=False, num_classes=84) # it is built by Pytorch 
        # The first layer is replaced to work for black and white images (1 input channel instead of 3)
        self.network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)

    def forward(self, xb):
        return torch.log_softmax(self.network(xb), dim=1) # the network output is the log_softmax of the the probabilities



### ResNet34 is a nn.Module
class DFL_VGG16(nn.Module):
    def __init__(self, k = 1, nclass = 84): # used for k=1 (cf. paper) and 84 classes
        super(DFL_VGG16, self).__init__() # Handy Pytorch init

        self.k = k
        self.nclass = nclass
        
        # To understand the architecture, read the paper. 
        # k channels for one class, nclass is total classes, therefore k * nclass for conv6
        vgg16featuremap = torchvision.models.vgg16_bn(pretrained=True).features # we get the pretrained weights from VGG16
        conv1_conv4 = torch.nn.Sequential(*list(vgg16featuremap.children())[:-11])  # these weights are used for the first part (before it branches)
        conv5 = torch.nn.Sequential(*list(vgg16featuremap.children())[-11:]) # and also in the global stream
        conv6 = torch.nn.Conv2d(512, k * nclass, kernel_size = 1, stride = 1, padding = 0) 
        pool6 = torch.nn.MaxPool2d((56, 56), stride = (56, 56), return_indices = True) # patches

        # Feature extraction root
        self.conv1_conv4 = conv1_conv4

        # G-Stream
        self.conv5 = conv5
        self.cls5 = nn.Sequential(
            nn.Conv2d(512, 84, kernel_size=1, stride = 1, padding = 0),
            nn.BatchNorm2d(84),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
            )

        # P-Stream
        self.conv6 = conv6
        self.pool6 = pool6
        self.cls6 = nn.Sequential(
            nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
            nn.AdaptiveAvgPool2d((1,1)),
            )

        # Side-branch
        self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)

        # Adapt to our images
        self.to_RGB = nn.ConvTranspose2d(1, 3, stride=1, kernel_size=1 , padding=0 ,bias=False)
        self.correcting_size = torch.nn.functional.interpolate

    def forward(self, x):
        batchsize = x.size(0)
        
        right_sized = self.correcting_size(x,[448,448])
        RGB_version = self.to_RGB(right_sized)

        #print(' We want [batch, 3 , 448, 448] but we recieve : ', RGB_version.shape) 
        inter4 = self.conv1_conv4(RGB_version)

        # G-stream
        x_g = self.conv5(inter4)
        out1 = self.cls5(x_g)
        out1 = out1.view(batchsize, -1)

        # P-stream ,indices is for visualization
        x_p = self.conv6(inter4)
        x_p, _ = self.pool6(x_p)
        inter6 = x_p
        out2 = self.cls6(x_p)
        out2 = out2.view(batchsize, -1)

        # Side-branch
        inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
        out3 = self.cross_channel_pool(inter6)
        out3 = out3.view(batchsize, -1)
    
        return torch.log_softmax(out1 + out2 + 0.1 * out3, dim=1) 
