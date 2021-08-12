import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.functional as F
import sys
import os
import numpy as np
import cv2

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

RELUSLOPE = 0.2

# KITENET #
class kitenet(nn.Module):
    def __init__(self, config):
        super(kitenet, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]+1
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.N_Filters      = 64
        self.N_modules      = config["N_modules"]

        self.encoder1       = nn.Conv2d(self.in_channels, 32, 3, stride=1, padding=1)
        self.encoder2       = nn.Conv2d(32, 64, 3, stride=1, padding=1) 
        self.encoder3       = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        
        
        self.decoder1       = nn.Conv2d(128, 64, 3, stride=1, padding=1) 
        self.decoder2       = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder3       = nn.Conv2d(32, self.out_channels, 3, stride=1, padding=1)

        self.relu     = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)
        
    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(4,4), mode ='bilinear')
        else:
            X_MS_UP = X_MS

        x = torch.cat((X_MS_UP, X_PAN.unsqueeze(1)), dim=1)

        out = self.relu(F.interpolate(self.encoder1(x),scale_factor=(2,2),mode ='bilinear'))
        out = self.relu(F.interpolate(self.encoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = self.relu(F.interpolate(self.encoder3(out),scale_factor=(2,2),mode ='bilinear'))

        out = self.relu(F.max_pool2d(self.decoder1(out), 2, 2))
        out = self.relu(F.max_pool2d(self.decoder2(out), 2, 2))
        out = self.relu(F.max_pool2d(self.decoder3(out), 2, 2))
        
        out = out + X_MS_UP
        return out

# KITENET with skip connections #
class kitenetwithskold(nn.Module):
    def __init__(self, config):
        super(kitenetwithskold, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]+1
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.N_Filters      = 64
        self.N_modules      = config["N_modules"]

        filters = [32, 64, 128]

        self.encoder1 = nn.Conv2d(self.in_channels, filters[0], 3, stride=1, padding=1)  # b, 16, 10, 10
        self.ebn1     = nn.BatchNorm2d(filters[0])
        self.encoder2 = nn.Conv2d(filters[0], filters[1], 3, stride=1, padding=1)  # b, 8, 3, 3
        self.ebn2     = nn.BatchNorm2d(filters[1])
        self.encoder3 = nn.Conv2d(filters[1], filters[2], 3, stride=1, padding=1)
        self.ebn3     = nn.BatchNorm2d(filters[2])
        
        self.decoder1 = nn.Conv2d(filters[2], filters[1], 3, stride=1, padding=1)  # b, 1, 28, 28
        self.dbn1     = nn.BatchNorm2d(filters[1])
        self.decoder2 = nn.Conv2d(2*filters[1], filters[0], 3, stride=1, padding=1)
        self.dbn2     = nn.BatchNorm2d(filters[0])
        self.decoder3 = nn.Conv2d(2*filters[0], filters[0], 3, stride=1, padding=1)
        self.dbn3     = nn.BatchNorm2d(filters[0])

        self.final_conv =  nn.Conv2d(filters[0], self.out_channels, 1)

        self.relu     = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)
        
    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(4,4), mode ='bicubic')
        else:
            X_MS_UP = X_MS

        x = torch.cat((X_MS_UP, X_PAN.unsqueeze(1)), dim=1)

        out = self.relu(self.ebn1(F.interpolate(self.encoder1(x),scale_factor=(2,2),mode ='bicubic')))
        t1 = out
        out = self.relu(self.ebn2(F.interpolate(self.encoder2(out),scale_factor=(2,2),mode ='bicubic')))
        t2 = out
        out = self.relu(self.ebn3(F.interpolate(self.encoder3(out),scale_factor=(2,2),mode ='bicubic')))

        out = self.relu(self.dbn1(F.max_pool2d(self.decoder1(out), 2, 2)))
        out = torch.cat((out, t2), dim=1)
        out = self.relu(self.dbn2(F.max_pool2d(self.decoder2(out), 2, 2)))
        out = torch.cat((out, t1), dim=1)
        out = self.relu(self.dbn3(F.max_pool2d(self.decoder3(out), 2, 2)))

        out = self.final_conv(out)
        
        out = out + X_MS_UP
        return out

# KITENET with skip connections #
class kitenetwithsk(nn.Module):
    def __init__(self, config):
        super(kitenetwithsk, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]+1
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]

        filters = [32, 64, 128]

        # ENCODER FILTERS
        self.encoder1 = nn.Conv2d(self.in_channels, filters[0], 3, stride=1, padding=1)
        self.ebn1     = nn.BatchNorm2d(filters[0])
        self.encoder2 = nn.Conv2d(filters[0], filters[1], 3, stride=1, padding=1)
        self.ebn2     = nn.BatchNorm2d(filters[1])
        self.encoder3 = nn.Conv2d(filters[1], filters[2], 3, stride=1, padding=1)
        self.ebn3     = nn.BatchNorm2d(filters[2])

        # BOTTELENECK FILTERS
        self.endec_conv   = nn.Conv2d(filters[2], filters[2], 3, stride=1, padding=1)
        self.endec_bn     = nn.BatchNorm2d(filters[2])
        
        # DECODER FILTERS
        self.decoder1 = nn.Conv2d(2*filters[2], filters[1], 3, stride=1, padding=1)  # b, 1, 28, 28
        self.dbn1     = nn.BatchNorm2d(filters[1])
        self.decoder2 = nn.Conv2d(2*filters[1], filters[0], 3, stride=1, padding=1)
        self.dbn2     = nn.BatchNorm2d(filters[0])
        self.decoder3 = nn.Conv2d(2*filters[0], self.out_channels, 3, stride=1, padding=1)
        self.dbn3     = nn.BatchNorm2d(self.out_channels)

        # FINAL CONV LAYER
        self.final_conv =  nn.Conv2d(self.out_channels, self.out_channels, 1)

        # RELU
        self.relu     = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)
        
    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(4,4), mode ='bilinear')
        else:
            X_MS_UP = X_MS

        x = torch.cat((X_MS_UP, X_PAN.unsqueeze(1)), dim=1)

        # ENCODER
        out = self.relu(self.ebn1(self.encoder1(x)))
        t1  = out

        out = F.interpolate(out,scale_factor=(2,2),mode ='bilinear')
        out = self.relu(self.ebn2(self.encoder2(out)))
        t2  = out

        out = F.interpolate(out,scale_factor=(2,2),mode ='bilinear')
        out = self.relu(self.ebn3(self.encoder3(out)))
        t3  = out

        # BOTTLENECK
        out = F.interpolate(out,scale_factor=(2,2),mode ='bilinear')
        out = self.relu(self.endec_bn(self.endec_conv(out)))
        
        # DECODER
        out = F.max_pool2d(out, 2, 2)
        out = torch.cat((out, t3), dim=1)
        out = self.relu(self.dbn1(self.decoder1(out)))

        out = F.max_pool2d(out, 2, 2)
        out = torch.cat((out, t2), dim=1)
        out = self.relu(self.dbn2(self.decoder2(out)))

        out = F.max_pool2d(out, 2, 2)
        out = torch.cat((out, t1), dim=1)
        out = self.relu(self.dbn3(self.decoder3(out)))

        # OUTPUT CONV
        out = self.final_conv(out)
        
        # FINAL OUTPUT
        out = out + X_MS_UP

        output = {"pred": out}
        return output

# KIUNET #
class kiunet(nn.Module):
    def __init__(self, config):
        super(kiunet, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]+1
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.N_Filters      = 64
        self.N_modules      = config["N_modules"]
        
        self.encoder1 = nn.Conv2d(self.in_channels, 16, 3, stride=1, padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB 
        self.en1_bn = nn.BatchNorm2d(16)
        self.encoder2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)  
        self.en2_bn = nn.BatchNorm2d(32)
        self.encoder3=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.en3_bn = nn.BatchNorm2d(64)

        self.decoder1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)   
        self.de1_bn = nn.BatchNorm2d(32)
        self.decoder2 =   nn.Conv2d(32,16, 3, stride=1, padding=1)
        self.de2_bn = nn.BatchNorm2d(16)
        self.decoder3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.de3_bn = nn.BatchNorm2d(8)

        self.decoderf1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.def1_bn = nn.BatchNorm2d(32)
        self.decoderf2=   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.def2_bn = nn.BatchNorm2d(16)
        self.decoderf3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.def3_bn = nn.BatchNorm2d(8)

        self.encoderf1 =   nn.Conv2d(self.in_channels, 16, 3, stride=1, padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB 
        self.enf1_bn = nn.BatchNorm2d(16)
        self.encoderf2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.enf2_bn = nn.BatchNorm2d(32)
        self.encoderf3 =   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.enf3_bn = nn.BatchNorm2d(64)

        self.intere1_1 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.inte1_1bn = nn.BatchNorm2d(16)
        self.intere2_1 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.inte2_1bn = nn.BatchNorm2d(32)
        self.intere3_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.inte3_1bn = nn.BatchNorm2d(64)

        self.intere1_2 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.inte1_2bn = nn.BatchNorm2d(16)
        self.intere2_2 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.inte2_2bn = nn.BatchNorm2d(32)
        self.intere3_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.inte3_2bn = nn.BatchNorm2d(64)

        self.interd1_1 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.intd1_1bn = nn.BatchNorm2d(32)
        self.interd2_1 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.intd2_1bn = nn.BatchNorm2d(16)
        self.interd3_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.intd3_1bn = nn.BatchNorm2d(64)

        self.interd1_2 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.intd1_2bn = nn.BatchNorm2d(32)
        self.interd2_2 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.intd2_2bn = nn.BatchNorm2d(16)
        self.interd3_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.intd3_2bn = nn.BatchNorm2d(64)

        self.final = nn.Conv2d(8,self.out_channels,1,stride=1,padding=0)

        self.relu     = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)
        
    
    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(4,4), mode ='bilinear')
        else:
            X_MS_UP = X_MS

        x = torch.cat((X_MS_UP, X_PAN.unsqueeze(1)), dim=1)

        out = self.relu(self.en1_bn(F.max_pool2d(self.encoder1(x),2,2)))  #U-Net branch
        out1 = self.relu(self.enf1_bn(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear'))) #Ki-Net branch
        tmp = out
        out = torch.add(out,F.interpolate(self.relu(self.inte1_1bn(self.intere1_1(out1))),scale_factor=(0.25,0.25),mode ='bilinear')) #CRFB
        out1 = torch.add(out1,F.interpolate(self.relu(self.inte1_2bn(self.intere1_2(tmp))),scale_factor=(4,4),mode ='bilinear')) #CRFB

        u1 = out  #skip conn
        o1 = out1  #skip conn

        out = self.relu(self.en2_bn(F.max_pool2d(self.encoder2(out),2,2)))
        out1 = self.relu(self.enf2_bn(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear')))

        tmp = out
        out = torch.add(out,F.interpolate(self.relu(self.inte2_1bn(self.intere2_1(out1))),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(self.relu(self.inte2_2bn(self.intere2_2(tmp))),scale_factor=(16,16),mode ='bilinear'))
        u2 = out
        o2 = out1

        out = self.relu(self.en3_bn(F.max_pool2d(self.encoder3(out),2,2)))
        out1 = self.relu(self.enf3_bn(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear')))
        tmp = out
        out = torch.add(out,F.interpolate(self.relu(self.inte3_1bn(self.intere3_1(out1))),scale_factor=(0.015625,0.015625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(self.relu(self.inte3_2bn(self.intere3_2(tmp))),scale_factor=(64,64),mode ='bilinear'))
        
        ### End of encoder block

        ### Start Decoder
        
        out = self.relu(self.de1_bn(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear')))  #U-NET
        out1 = self.relu(self.def1_bn(F.max_pool2d(self.decoderf1(out1),2,2))) #Ki-NET

        tmp = out
        out = torch.add(out,F.interpolate(self.relu(self.intd1_1bn(self.interd1_1(out1))),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(self.relu(self.intd1_2bn(self.interd1_2(tmp))),scale_factor=(16,16),mode ='bilinear'))
        out = torch.add(out,u2)  #skip conn
        out1 = torch.add(out1,o2)  #skip conn

        out = self.relu(self.de2_bn(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear')))
        out1 = self.relu(self.def2_bn(F.max_pool2d(self.decoderf2(out1),2,2)))
        tmp = out
        out = torch.add(out,F.interpolate(self.relu(self.intd2_1bn(self.interd2_1(out1))),scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(self.relu(self.intd2_2bn(self.interd2_2(tmp))),scale_factor=(4,4),mode ='bilinear'))

        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = self.relu(self.de3_bn(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear')))
        out1 = self.relu(self.def3_bn(F.max_pool2d(self.decoderf3(out1),2,2)))

        out = torch.add(out,out1) # fusion of both branches

        out = self.final(out)

        out = out + X_MS_UP
        
        return out

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

# KITENET with skip connections #
class attentionkitenet(nn.Module):
    def __init__(self, config):
        super(attentionkitenet, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]+1
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.N_Filters      = 64
        self.N_modules      = config["N_modules"]

        self.encoder1 = nn.Conv2d(self.in_channels, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        
        self.decoder1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder2 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(64, self.out_channels, 3, stride=1, padding=1)

        self.relu     = nn.LeakyReLU(negative_slope=RELUSLOPE, inplace=True)

        self.Attention1 = Attention_block(64, 64, 32)
        self.Attention2 = Attention_block(32, 32, 16)
        
    def forward(self, X_MS, X_PAN):
        if not self.is_DHP_MS:
            X_MS_UP = F.interpolate(X_MS, scale_factor=(4,4), mode ='bilinear')
        else:
            X_MS_UP = X_MS

        x = torch.cat((X_MS_UP, X_PAN.unsqueeze(1)), dim=1)

        out = self.relu(F.interpolate(self.encoder1(x),scale_factor=(2,2),mode ='bilinear'))
        t1 = out
        out = self.relu(F.interpolate(self.encoder2(out),scale_factor=(2,2),mode ='bilinear'))
        t2 = out
        out = self.relu(F.interpolate(self.encoder3(out),scale_factor=(2,2),mode ='bilinear'))

        out     = self.relu(F.max_pool2d(self.decoder1(out), 2, 2))
        out_a   = self.Attention1(g=out, x=t2)
        out     = torch.cat((out_a, out), dim=1)

        out = self.relu(F.max_pool2d(self.decoder2(out), 2, 2))
        out_a   = self.Attention2(g=out, x=t1)
        out     = torch.cat((out_a, out), dim=1)

        out = self.relu(F.max_pool2d(self.decoder3(out), 2, 2))
        
        out = out + X_MS_UP
        return out