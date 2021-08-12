from math import ceil
import torch
import torch.nn.functional as F
from torch import nn
import math
import kornia
from kornia import laplacian

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

RELUSLOPE = 0.1

# Spatial Graph Reasoning ...
class Spatial_GR(nn.Module):
    def __init__(self, plane):
        super(Spatial_GR, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))
        self.out_bn = nn.BatchNorm2d(plane)
        self.out_relu = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)

    def forward(self, x):
        # b, c, h, w = x.size()
        #Spatial GR
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        
        #Final output
        out = self.out_relu(self.out(AVW) + x)
        #out = self.out(AVW)
        return out

# Spectral attention module ...
class Speactral_Attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Speactral_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Residual Spectral-Spatial Graph Reasoning Block
class Res_SSGR(nn.Module):
    def __init__(self, in_channel):
        super(Res_SSGR, self).__init__()
        self.spectral_atten     = Speactral_Attention(channel=in_channel, reduction=8)
        self.spatial_GR         = Spatial_GR(plane=in_channel) 
        self.our_relu           = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True) 
    def forward(self, x):
        x_s     = self.spectral_atten(x)
        out     = self.spatial_GR(x_s)
        #out     = x_ss + x
        #out     = self.our_relu(out)
        return out

class _ConvBnReLU_SGR(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, LeakyReLU, and Spatial Graph Reasoning.
    """

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU_SGR, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=1 - 0.999))

        if relu:
            self.add_module("relu", nn.LeakyReLU(negative_slope=RELUSLOPE))
        
        self.add_module("sgr", Spatial_GR(out_ch))

class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU_SGR(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU_SGR(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        #self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)

class AGPP_Block(nn.Module):
    def __init__(self, in_channel):
        super(AGPP_Block, self).__init__()
        #rates = [2, 4, 8]
        rates = [2, 4, 8]
        self.spectral_atten     = Speactral_Attention(channel=in_channel, reduction=8)
        self.AGPP               = _ASPP(in_ch=in_channel, out_ch=in_channel, rates=rates)
        self.out_conv           =  nn.Conv2d(in_channels=in_channel*(1+len(rates)), out_channels=in_channel, kernel_size=1)
        self.out_relu           = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True) 
    def forward(self, x):
        x_s     = self.spectral_atten(x)
        x_ss    = self.out_conv(self.AGPP(x_s))
        out     = x_ss + x
        out     = self.out_relu(out)
        return out

## DHP-GR ###
class DHP_SSGR(nn.Module):
    def __init__(self, config):
        super(DHP_SSGR, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.N_Filters      = 64
        self.N_modules      = config["N_modules"]

        # FEN Layer
        self.FEN        = nn.Conv2d(in_channels=2*(self.in_channels+1), out_channels=self.N_Filters, kernel_size=3, padding=1)
        self.FEN_bn     = nn.BatchNorm2d(self.N_Filters)
        self.FEN_relu   = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)
        
        # CSA RESBLOCKS
        modules=[]
        for i in range(self.N_modules):
            modules.append(AGPP_Block(self.N_Filters))
        self.Spectral_Spatial_GR = nn.ModuleList(modules)
        
        # RNN layer
        self.RRN        = nn.Conv2d(in_channels=self.N_Filters, out_channels=self.out_channels, kernel_size=3, padding=1)
        self.RRN_bn     = nn.BatchNorm2d(self.out_channels)
        self.RRN_relu   = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)
        

    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(4,4), mode ='bilinear')
        else:
            X_MS_UP = X_MS

        # Concatenating the generated H_UP with P
        x_lp = torch.cat((X_MS_UP, X_PAN.unsqueeze(1)), dim=1)
        x_hp = torch.cat((laplacian(X_MS_UP, 3), laplacian(X_PAN.unsqueeze(1), 3)), dim=1)
        x = torch.cat((x_lp, x_hp), dim=1)

        # FEN
        x = self.FEN(x)
        x = self.FEN_bn(x)
        x = self.FEN_relu(x)

        # DARN
        for i in range(self.N_modules):
            x = self.Spectral_Spatial_GR[i](x)

        # RRN
        x = self.RRN(x)
        #x = self.RRN_relu(x)

        # Final output
        x = x + X_MS_UP
        return x

## DHP-GRV2 ###
class DHP_SSGRV2(nn.Module):
    def __init__(self, config):
        super(DHP_SSGRV2, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.N_Filters      = 64
        self.N_modules      = config["N_modules"]

        # FEN Layer
        self.FEN        = nn.Conv2d(in_channels=2*(self.in_channels+1), out_channels=self.N_Filters, kernel_size=3, padding=1)
        self.FEN_bn     = nn.BatchNorm2d(self.N_Filters)
        self.FEN_relu   = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)
        
        # CSA RESBLOCKS
        modules=[]
        for i in range(self.N_modules):
            modules.append(Res_SSGR(self.N_Filters))
        self.Spectral_Spatial_GR = nn.ModuleList(modules)
        
        # RNN layer
        self.RRN        = nn.Conv2d(in_channels=self.N_Filters, out_channels=self.out_channels, kernel_size=3, padding=1)
        self.RRN_bn     = nn.BatchNorm2d(self.out_channels)
        self.RRN_relu   = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)
        

    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(4,4), mode ='bilinear')
        else:
            X_MS_UP = X_MS

        # Concatenating the generated H_UP with P
        x_lp = torch.cat((X_MS_UP, X_PAN.unsqueeze(1)), dim=1)
        x_hp = torch.cat((laplacian(X_MS_UP, 3), laplacian(X_PAN.unsqueeze(1), 3)), dim=1)
        x = torch.cat((x_lp, x_hp), dim=1)

        # FEN
        x = self.FEN(x)
        x = self.FEN_bn(x)
        x = self.FEN_relu(x)

        # DARN
        for i in range(self.N_modules):
            x = self.Spectral_Spatial_GR[i](x)

        # RRN
        x = self.RRN(x)
        #x = self.RRN_relu(x)

        # Final output
        x = x + X_MS_UP
        return x

## DHP-GRV2 ###
class DHP_SSGRV3(nn.Module):
    def __init__(self, config):
        super(DHP_SSGRV3, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.N_Filters      = 64
        self.N_modules      = config["N_modules"]

        # FEN Layer
        self.FEN1        = nn.Conv2d(in_channels=1, out_channels=self.N_Filters, kernel_size=7, padding=3)
        self.FEN_bn1     = nn.BatchNorm2d(self.N_Filters)
        self.FEN_relu1   = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)

        self.FEN2        = nn.Conv2d(in_channels=self.in_channels, out_channels=self.N_Filters, kernel_size=7, padding=3)
        self.FEN_bn2     = nn.BatchNorm2d(self.N_Filters)
        self.FEN_relu2   = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)

        self.FEN_bn3     = nn.BatchNorm2d(self.N_Filters)
        
        # CSA RESBLOCKS
        modules=[]
        for i in range(self.N_modules):
            modules.append(Res_SSGR(self.N_Filters))
        self.Spectral_Spatial_GR = nn.ModuleList(modules)
        
        # RNN layer
        self.RRN        = nn.Conv2d(in_channels=self.N_Filters, out_channels=self.out_channels, kernel_size=1, padding=0)
        

    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(4,4), mode ='bilinear')
        else:
            X_MS_UP = X_MS

        # FEN Layer
        x1 = self.FEN_relu1(self.FEN_bn1(self.FEN1(X_PAN.unsqueeze(1))))
        x2 = self.FEN_relu2(self.FEN_bn2(self.FEN2(X_MS_UP)))

        # Taking the High-Pass Output
        x = self.FEN_bn3(x2-x1)

        # DARN
        for i in range(self.N_modules):
            x = self.Spectral_Spatial_GR[i](x)

        # RRN
        x = self.RRN(x)

        # Final output
        x = x + X_MS_UP
        return x

## DHP-GRV2 ###
class DHP_SSGRV4(nn.Module):
    def __init__(self, config):
        super(DHP_SSGRV4, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.N_Filters      = 64
        self.N_modules      = config["N_modules"]

        # FEN Layer
        self.FEN1        = nn.Conv2d(in_channels=1, out_channels=self.N_Filters, kernel_size=7, padding=3)
        self.FEN_bn1     = nn.BatchNorm2d(self.N_Filters)
        self.FEN_relu1   = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)

        self.FEN2        = nn.Conv2d(in_channels=self.in_channels, out_channels=self.N_Filters, kernel_size=7, padding=3)
        self.FEN_bn2     = nn.BatchNorm2d(self.N_Filters)
        self.FEN_relu2   = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)

        self.FEN_bn3     = nn.BatchNorm2d(self.N_Filters)
        
        # CSA RESBLOCKS
        modules=[]
        for i in range(self.N_modules):
            modules.append(Res_SSGR(self.N_Filters))
        self.Spectral_Spatial_GR = nn.ModuleList(modules)
        
        # RNN layer
        self.RRN        = nn.Conv2d(in_channels=self.N_Filters, out_channels=self.out_channels, kernel_size=1, padding=0)
        

    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(4,4), mode ='bilinear')
        else:
            X_MS_UP = X_MS

        # FEN Layer
        x1 = self.FEN_relu1(self.FEN_bn1(self.FEN1(X_PAN.unsqueeze(1))))
        x2 = self.FEN_relu2(self.FEN_bn2(self.FEN2(X_MS_UP)))

        # Taking the High-Pass Output
        x = self.FEN_bn3(x2-x1)

        # DARN
        for i in range(self.N_modules):
            x = self.Spectral_Spatial_GR[i](x)

        # RRN
        x = self.RRN(x)

        # Final output
        x = x + X_MS_UP
        return x
# class DHP_SSGR(nn.Module):
#     def __init__(self, config):
#         super(DHP_SSGR, self).__init__()
#         self.is_DHP_MS      = config["is_DHP_MS"]
#         self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
#         self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
#         self.N_Filters      = 64
#         self.N_modules      = config["N_modules"]

#         # FEN Layer
#         self.FEN1        = nn.Conv2d(in_channels=1, out_channels=self.N_Filters, kernel_size=3, padding=1)
#         self.FEN_bn1     = nn.BatchNorm2d(self.N_Filters)
#         #self.FEN_relu1   = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)
        
#         self.FEN2        = nn.Conv2d(in_channels= self.in_channels, out_channels=self.N_Filters, kernel_size=3, padding=1)
#         self.FEN_bn2     = nn.BatchNorm2d(self.N_Filters)
#         #self.FEN_relu1   = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)
        
#         # CSA RESBLOCKS
#         modules=[]
#         for i in range(self.N_modules):
#             modules.append(AGPP_Block(self.N_Filters))
#         self.Spectral_Spatial_GR = nn.ModuleList(modules)
        
#         # RNN layer
#         self.RRN        = nn.Conv2d(in_channels=self.N_Filters, out_channels=self.out_channels, kernel_size=3, padding=1)
#         self.RRN_bn     = nn.BatchNorm2d(self.out_channels)
#         self.RRN_relu   = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)
        

#     def forward(self, X_MS, X_PAN):
#         if not self.is_DHP_MS:
#             X_MS_UP = F.interpolate(X_MS, scale_factor=(4,4), mode ='bilinear')
#         else:
#             X_MS_UP = X_MS

#         # FEN on PAN Image
#         x1 = self.FEN1(X_PAN.unsqueeze(1))
#         x1 = self.FEN_bn1(x1)

#         # FEN on Multispectral Image
#         x2 = self.FEN2(X_MS_UP)
#         x2 = self.FEN_bn2(x2)

#         # High pass filtering
#         x = x1-x2

#         # DARN
#         for i in range(self.N_modules):
#             x = self.Spectral_Spatial_GR[i](x)

#         # RRN
#         x = self.RRN(x)
#         #x = self.RRN_relu(x)

#         # Final output
#         x = x + X_MS_UP
#         return x