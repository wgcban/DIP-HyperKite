from math import ceil
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.functional as F

### HyperPNN3 IMPLEMENTATION ###
class HyperPNN(nn.Module):
    def __init__(self, config):
        super(HyperPNN, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.factor         = config[config["train_dataset"]]["factor"]

        self.mid_channels = 64
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=self.mid_channels+1, out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=1)
        self.conv7 = nn.Conv2d(in_channels=self.mid_channels, out_channels=self.out_channels, kernel_size=1)
        self.Sigmoid = nn.Sigmoid()
    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor,self.factor),mode ='bilinear')
        else:
            X_MS_UP = X_MS
        x = F.relu(self.conv1(X_MS_UP))
        x = F.relu(self.conv2(x))
        x = torch.cat((x, X_PAN.unsqueeze(1)), dim=1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        x = x+X_MS_UP

        output = {  "pred": x}
        return output