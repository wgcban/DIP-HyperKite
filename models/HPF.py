from math import ceil
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.functional as F
import math
import kornia
from kornia import laplacian

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

RELUSLOPE = 0.1

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)

class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning

class RDN(nn.Module):
    def __init__(self, config):
        super(RDN, self).__init__()
        # Parameters
        self.is_DHP_MS      = config["is_DHP_MS"]
        num_channels = config[config["train_dataset"]]["spectral_bands"]+1
        out_channels   = config[config["train_dataset"]]["spectral_bands"]
        num_features = 32
        growth_rate = 32
        num_blocks = 4      #Number of residual dence blocks
        num_layers = 4      #Number of dense layers in a dense block

        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe1_bn = nn.BatchNorm2d(num_features)
        self.sfe1_relu = nn.LeakyReLU(RELUSLOPE, True)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2_bn = nn.BatchNorm2d(num_features)
        self.sfe2_relu = nn.LeakyReLU(RELUSLOPE, True)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, out_channels, kernel_size=3, padding=3 // 2)
        )

        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(4,4), mode ='bilinear')
        else:
            X_MS_UP = X_MS
        
        print(X_MS_UP.shape)
        x = torch.cat((X_MS_UP, X_PAN.unsqueeze(1)), dim=1)
        print(x.shape)

        sfe1 = self.sfe1_relu(self.sfe1_bn(self.sfe1(x)))
        sfe2 = self.sfe2_relu(self.sfe2_bn(self.sfe2(sfe1)))

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + X_MS_UP  # global residual learning
        return x

class HPF(nn.Module):
    def __init__(self, config):
        super(HPF, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.N_Filters      = 64
        self.N_modules      = config["N_modules"]

        self.conv1 = nn.Conv2d(in_channels=1, out_channels= self.N_Filters, kernel_size=7, padding=3)
        self.conv1_bn = nn.BatchNorm2d(self.N_Filters)
        self.conv1_relu = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)

        self.conv2 = nn.Conv2d(in_channels=self.N_Filters, out_channels= self.N_Filters, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(self.N_Filters)
        self.conv2_relu = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)

        self.conv3 = nn.Conv2d(in_channels=self.N_Filters, out_channels=self.out_channels, kernel_size=1)
        
    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(4,4), mode ='bilinear')
        else:
            X_MS_UP = X_MS

        # FEN Layer
        x = X_PAN.unsqueeze(1)

        x = self.conv1_relu(self.conv1_bn(self.conv1(x)))
        x = self.conv2_relu(self.conv2_bn(self.conv2(x)))
        x = self.conv3(x)
        
        x = x + X_MS_UP
        return x
