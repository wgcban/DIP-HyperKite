from math import ceil
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.functional as F
import math


# Updated PAN prediction network...
class Predict_PAN(nn.Module):
    def __init__(self, spectral_bands):
        super(Predict_PAN, self).__init__()
        r = 2
        self.AvgPool    = nn.AdaptiveAvgPool2d(1)
        self.conv1      = nn.Conv2d(in_channels=spectral_bands, out_channels=int(spectral_bands/r), kernel_size=1)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = nn.Conv2d(in_channels=int(spectral_bands/r), out_channels=spectral_bands, kernel_size=1)
        self.SoftMax    = nn.Softmax(dim=1)
        

    def forward(self, net_output, mode="GAP"):
        if mode=="GAP":
            R           = self.SoftMax(self.conv2(self.relu(self.conv1(self.AvgPool(net_output)))))
            PAN_pred    = torch.sum(net_output*R.expand_as(net_output), dim=1)
            return PAN_pred
        elif mode=="MEAN":
            PAN_pred    = torch.mean(net_output, dim=1)
            return PAN_pred
        