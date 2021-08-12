# Spectral Transformer
# Author: Chaminda Bandara
# Date: 07/06/2021

import torch
import torch.nn.functional as F
from torch import nn
from scipy.io import savemat

LOSS_TP = nn.L1Loss()

EPS = 1e-10

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        
    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out

class SFE(nn.Module):
    def __init__(self, in_feats, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(in_feats, n_feats)
        
        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats, 
                res_scale=res_scale))
            
        self.conv_tail = conv3x3(n_feats, n_feats)
        
    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class CSFI2(nn.Module):
    def __init__(self, n_feats):
        super(CSFI2, self).__init__()
        self.conv12 = conv1x1(n_feats, int(n_feats/2))
        self.conv21 = conv3x3(int(n_feats/2), n_feats, 2)

        self.conv_merge1 = conv3x3(n_feats*2, n_feats)
        self.conv_merge2 = conv3x3(n_feats, int(n_feats/2))

    def forward(self, x1, x2):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x21 = F.relu(self.conv21(x2))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12), dim=1) ))

        return x1, x2


class CSFI3(nn.Module):
    def __init__(self, n_feats):
        super(CSFI3, self).__init__()
        n_feats1 = n_feats
        self.conv12 = conv1x1(n_feats1, n_feats1)
        self.conv13 = conv1x1(n_feats1, n_feats1)

        n_feats2 = int(n_feats/2)
        self.conv21 = conv3x3(n_feats2, n_feats2, 2)
        self.conv23 = conv1x1(n_feats2, n_feats2)

        n_feats3 = int(n_feats/4)
        self.conv31_1 = conv3x3(n_feats3, n_feats3, 2)
        self.conv31_2 = conv3x3(n_feats3, n_feats3, 2)
        self.conv32 = conv3x3(n_feats3, n_feats3, 2)

        self.conv_merge1 = conv3x3(n_feats1*3, n_feats1)
        self.conv_merge2 = conv3x3(n_feats2*3, n_feats2)
        self.conv_merge3 = conv3x3(n_feats3*3, n_feats3)

    def forward(self, x1, x2, x3):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))

        x21 = F.relu(self.conv21(x2))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x31 = F.relu(self.conv31_1(x3))
        x31 = F.relu(self.conv31_2(x31))
        x32 = F.relu(self.conv32(x3))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21, x31), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12, x32), dim=1) ))
        x3 = F.relu(self.conv_merge3( torch.cat((x3, x13, x23), dim=1) ))
        
        return x1, x2, x3


class MergeTail(nn.Module):
    def __init__(self, n_feats, out_channels):
        super(MergeTail, self).__init__()
        self.conv13 = conv1x1(n_feats, int(n_feats/4))
        self.conv23 = conv1x1(int(n_feats/2), int(n_feats/4))
        self.conv_merge = conv3x3(3*int(n_feats/4), out_channels)
        self.conv_tail1 = conv3x3(out_channels, out_channels)
        self.conv_tail2 = conv1x1(n_feats//2, 3)

    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge( torch.cat((x3, x13, x23), dim=1) ))
        x = self.conv_tail1(x)
        #x = self.conv_tail2(x)
        
        return x

# This function implements the learnable spectral feature extractor (abreviated as LSFE)
# Input:    Hyperspectral or PAN image
# Outputs:  out1 = features at original resolution, out2 = features at original resolution/2, out3 = features at original resolution/4
class LFE(nn.Module):
    def __init__(self, in_channels):
        super(LFE, self).__init__()
        #Define number of input channels
        self.in_channels = in_channels
        #First level convolutions
        self.conv_64_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, padding=3)
        self.conv_64_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        #Second level convolutions
        self.conv_128_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        #Third level convolutions
        self.conv_256_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv_256_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        #Max pooling
        self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #LeakyReLU 
        self.LeakyReLU  = nn.LeakyReLU(negative_slope=0.0)
    def forward(self, x):
        #First level outputs
        out1    = self.LeakyReLU(self.conv_64_2(self.LeakyReLU(self.conv_64_1(x))))
        #Second level outputs
        out1_mp = self.MaxPool2x2(out1)
        out2    = self.LeakyReLU(self.conv_128_2(self.LeakyReLU(self.conv_128_1(out1_mp))))
        #Third level outputs
        out2_mp = self.MaxPool2x2(out2)
        out3    = self.conv_256_2(self.LeakyReLU(self.conv_256_1(out2_mp))) #Without relu output - faster convergence observed.
        return out1, out2, out3

#This function implements the Spectral Feature Relavance Embedding  (SFRE)
#Input:     K, Q, and V
#Outputs:   cross-correlation between each feature map
class RE(nn.Module):
    def __init__(self):
        super(RE, self).__init__()
    def forward(self, K, Q):
        #Reading input image size
        b, c, h, w = K.shape
        #Reshaping K and Q
        K_reshaped = K.view(b, c, h*w)
        Q_reshaped  = Q.view(b, c, h*w)
        #Calculating each channel mean over pixel dimension
        K_mean      = torch.mean(K_reshaped, dim=2).unsqueeze(2)
        Q_mean      = torch.mean(Q_reshaped, dim=2).unsqueeze(2)
        #Calculating cross-correlation matrix
        CC = torch.matmul((K_reshaped-K_mean), (Q_reshaped-Q_mean).permute(0, 2, 1))
        return CC

#This function implements the Hard Attention on Features (HAF)
#Input:     K, Q, and V
#Outputs:   cross-correlation between each feature map
class TS_Hard(nn.Module):
    def __init__(self):
        super(TS_Hard, self).__init__()
        self.RE  = RE()
    
    def forward(self, V, K, Q):
        #Calculating cross-correlation between K and Q => (N_k, N_q)
        CC      = self.RE(K, Q) #N_k, N_q
        S, H    = torch.max(CC, dim=1)
        
        T = torch.cat([ torch.index_select(v, 0, h).unsqueeze(0) for v, h in zip(V, H) ])
        return T, S

#This function implements the Soft Attention on Features (HAF)
#Input:     K, Q, and V
#Outputs:   cross-correlation between each feature map
class TS_Soft(nn.Module):
    def __init__(self):
        super(TS_Soft, self).__init__()
        self.RE         = RE()
        self.softmax    = nn.Softmax(dim=2)
    
    def forward(self, V, K, Q):
        #Calculating cross-correlation between K and Q => (N_k, N_q)
        CC              = self.RE(K, Q) #b, N_k, N_q
        S, H            = torch.max(CC, dim=1)
        CC_normalized   = self.softmax(CC)
        
        N_Q = CC_normalized.shape[2]

        T = torch.cat([ torch.mean(V*CC_normalized[:, :, q].unsqueeze(2).unsqueeze(3), 1).unsqueeze(1) for q in range(N_Q) ], dim=1)
        return T, S

#######################################
# Hyperspectral Transformer (HSIT) ####
#         Initial Training         ####
#######################################
# We pre-train this model first and then train the above model with pre-trained weights
class HyperTransformerPre(nn.Module):
    def __init__(self, config):
        super(HyperTransformerPre, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.factor         = config[config["train_dataset"]]["factor"]

        self.num_res_blocks = [16, 1, 1, 1, 4]
        self.n_feats        = 256
        self.res_scale      = 1

        self.LFE_HSI    = LFE(in_channels=self.in_channels)
        self.LFE_PAN    = LFE(in_channels=1)
        self.TS         = TS_Hard()
        self.SFE        = SFE(self.in_channels, self.num_res_blocks[0], self.n_feats, self.res_scale)

        ###############
        ### stage11 ###
        ###############
        self.conv11_head    = conv3x3(2*self.n_feats, self.n_feats)
        self.conv12         = conv3x3(self.n_feats, self.n_feats*2)
        self.ps12           = nn.PixelShuffle(2)

        ###############
        ### stage22 ###
        ###############
        self.conv22_head    = conv3x3(2*int(self.n_feats/2), int(self.n_feats/2))
        self.conv23         = conv3x3(int(self.n_feats/2), self.n_feats)
        self.ps23           = nn.PixelShuffle(2)

        ###############
        ### stage33 ###
        ###############
        self.conv33_head    = conv3x3(2*int(self.n_feats/4), int(self.n_feats/4))
        

        ##############
        ### FINAL ####
        ##############
        self.final_conv     = nn.Conv2d(in_channels=self.n_feats+int(self.n_feats/2)+int(self.n_feats/4), out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, X_MS, X_PAN):
        with torch.no_grad():
            if not self.is_DHP_MS:
                X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor,self.factor),mode ='bicubic')

            else:
                X_MS_UP = X_MS
            
            # Generating PAN, and PAN (UD) images
            X_PAN   = X_PAN.unsqueeze(dim=1)
            PAN_D   = F.interpolate(X_PAN, scale_factor=(1/self.factor, 1/self.factor), mode ='bilinear')
            PAN_UD  = F.interpolate(PAN_D, scale_factor=(self.factor, self.factor), mode ='bilinear')

        #Extracting T and S at multiple-scales
        #lv1 = LxW, lv2 = (L/2)x(w/2), lv3=(L/4, W/4)
        V_lv1, V_lv2, V_lv3 = self.LFE_PAN(X_PAN)
        K_lv1, K_lv2, K_lv3 = self.LFE_PAN(PAN_UD)
        Q_lv1, Q_lv2, Q_lv3 = self.LFE_HSI(X_MS_UP)

        T_lv3, S_lv3  = self.TS(V_lv3, K_lv3, Q_lv3)
        T_lv2, S_lv2  = self.TS(V_lv2, K_lv2, Q_lv2)
        T_lv1, S_lv1  = self.TS(V_lv1, K_lv1, Q_lv1)

        #Shallow Feature Extraction (SFE)
        x = self.SFE(X_MS)

        #####################################
        #### stage11: (L/4, W/4) scale ######
        #####################################
        x11 = x
        #HyperTransformer at (L/4, W/4) scale
        x11_res = x11
        x11_res = torch.cat((x11_res, T_lv3), dim=1)
        x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
        S_lv3   = S_lv3.view(S_lv3.shape[0], S_lv3.shape[1], 1, 1)
        x11_res = x11_res * S_lv3.expand_as(x11_res)
        x11     = x11 + x11_res

        #####################################
        #### stage22: (L/2, W/2) scale ######
        #####################################
        x22 = self.conv12(x11)
        x22 = F.relu(self.ps12(x22))
        #HyperTransformer at (L/2, W/2) scale
        x22_res = x22
        x22_res = torch.cat((x22_res, T_lv2), dim=1)
        x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
        S_lv2   = S_lv2.view(S_lv2.shape[0], S_lv2.shape[1], 1, 1)
        x22_res = x22_res * S_lv2.expand_as(x22_res)
        x22     = x22 + x22_res

        #####################################
        ###### stage22: (L, W) scale ########
        #####################################
        x33 = self.conv23(x22)
        x33 = F.relu(self.ps23(x33))
        #HyperTransformer at (L, W) scale
        x33_res = x33
        x33_res = torch.cat((x33_res, T_lv1), dim=1)
        x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
        S_lv1   = S_lv1.view(S_lv1.shape[0], S_lv1.shape[1], 1, 1)
        x33_res = x33_res * S_lv1.expand_as(x33_res)
        x33     = x33 + x33_res

        #####################################
        ############ Feature Pyramid ########
        #####################################
        x11_up  = F.interpolate(x11, scale_factor=4, mode='bicubic')
        x22_up  = F.interpolate(x22, scale_factor=2, mode='bicubic')
        xF      = torch.cat((x11_up, x22_up, x33), dim=1)
        
        #####################################
        ####  Final convolution   ###########
        #####################################
        x      = self.final_conv(xF)
        return x


#######################################
# Hyperspectral Transformer (HSIT) ####
#######################################
class HyperTransformer(nn.Module):
    def __init__(self, config):
        super(HyperTransformer, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.factor         = config[config["train_dataset"]]["factor"]

        self.num_res_blocks = [16, 1, 1, 1, 4]
        self.n_feats        = 256
        self.res_scale      = 1

        self.LFE_HSI    = LFE(in_channels=self.in_channels)
        self.LFE_PAN    = LFE(in_channels=1)
        self.TS         = TS_Hard()
        self.SFE        = SFE(self.in_channels, self.num_res_blocks[0], self.n_feats, self.res_scale)

        ###############
        ### stage11 ###
        ###############
        self.conv11_head    = conv3x3(2*self.n_feats, self.n_feats)
        self.conv12         = conv3x3(self.n_feats, self.n_feats*2)
        self.ps12           = nn.PixelShuffle(2)
        #Residial blocks
        self.RB11           = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels=self.n_feats, out_channels=self.n_feats,
                res_scale=self.res_scale))
        self.conv11_tail = conv3x3(self.n_feats, self.n_feats)

        ###############
        ### stage22 ###
        ###############
        self.conv22_head    = conv3x3(2*int(self.n_feats/2), int(self.n_feats/2))
        self.conv23         = conv3x3(int(self.n_feats/2), self.n_feats)
        self.ps23           = nn.PixelShuffle(2)
        #Residual blocks
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB22.append(ResBlock(in_channels=int(self.n_feats/2), out_channels=int(self.n_feats/2), res_scale=self.res_scale))
        self.conv22_tail = conv3x3(int(self.n_feats/2), int(self.n_feats/2))

        ###############
        ### stage33 ###
        ###############
        self.conv33_head    = conv3x3(2*int(self.n_feats/4), int(self.n_feats/4))
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB33.append(ResBlock(in_channels=int(self.n_feats/4), out_channels=int(self.n_feats/4), res_scale=self.res_scale))
        self.conv33_tail = conv3x3(int(self.n_feats/4), int(self.n_feats/4))

        ##############
        ### FINAL ####
        ##############
        self.final_conv     = nn.Conv2d(in_channels=self.n_feats+int(self.n_feats/2)+int(self.n_feats/4), out_channels=self.out_channels, kernel_size=3, padding=1)
        self.RBF = nn.ModuleList()
        for i in range(self.num_res_blocks[4]):
            self.RBF.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
        self.convF_tail = conv3x3(self.out_channels, self.out_channels)



    def forward(self, X_MS, X_PAN):
        with torch.no_grad():
            if not self.is_DHP_MS:
                X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor,self.factor),mode ='bicubic')

            else:
                X_MS_UP = X_MS
            
            # Generating PAN, and PAN (UD) images
            X_PAN   = X_PAN.unsqueeze(dim=1)
            PAN_D   = F.interpolate(X_PAN, scale_factor=(1/self.factor, 1/self.factor), mode ='bilinear')
            PAN_UD  = F.interpolate(PAN_D, scale_factor=(self.factor, self.factor), mode ='bilinear')

        #Extracting T and S at multiple-scales
        #lv1 = LxW, lv2 = (L/2)x(w/2), lv3=(L/4, W/4)
        V_lv1, V_lv2, V_lv3 = self.LFE_PAN(X_PAN)
        K_lv1, K_lv2, K_lv3 = self.LFE_PAN(PAN_UD)
        Q_lv1, Q_lv2, Q_lv3 = self.LFE_HSI(X_MS_UP)

        T_lv3, S_lv3  = self.TS(V_lv3, K_lv3, Q_lv3)
        T_lv2, S_lv2  = self.TS(V_lv2, K_lv2, Q_lv2)
        T_lv1, S_lv1  = self.TS(V_lv1, K_lv1, Q_lv1)

        #Save feature maps for illustration purpose
        # feature_dic={}
        # f = 1e3
        # feature_dic.update({"V": f*V_lv3.detach().cpu().numpy()})
        # feature_dic.update({"K": f*K_lv3.detach().cpu().numpy()})
        # feature_dic.update({"Q": f*Q_lv3.detach().cpu().numpy()})
        # feature_dic.update({"T": f*T_lv3.detach().cpu().numpy()})
        # savemat("/home/lidan/Dropbox/Hyperspectral/HyperTransformer/feature_visualization_pavia/features_lv3.mat", feature_dic)
        # exit()

        #Shallow Feature Extraction (SFE)
        x = self.SFE(X_MS)

        #####################################
        #### stage1: (L/4, W/4) scale ######
        #####################################
        x11 = x
        #HyperTransformer at (L/4, W/4) scale
        x11_res = x11
        x11_res = torch.cat((x11_res, T_lv3), dim=1)
        x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
        S_lv3   = S_lv3.view(S_lv3.shape[0], S_lv3.shape[1], 1, 1)
        x11_res = x11_res * S_lv3.expand_as(x11_res)
        x11     = x11 + x11_res
        #Residial learning
        x11_res = x11
        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res

        #####################################
        #### stage2: (L/2, W/2) scale ######
        #####################################
        x22 = self.conv12(x11)
        x22 = F.relu(self.ps12(x22))
        #HyperTransformer at (L/2, W/2) scale
        x22_res = x22
        x22_res = torch.cat((x22_res, T_lv2), dim=1)
        x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
        S_lv2   = S_lv2.view(S_lv2.shape[0], S_lv2.shape[1], 1, 1)
        x22_res = x22_res * S_lv2.expand_as(x22_res)
        x22     = x22 + x22_res
        #Residial learning
        x22_res = x22
        for i in range(self.num_res_blocks[2]):
            x22_res = self.RB22[i](x22_res)
        x22_res = self.conv22_tail(x22_res)
        x22 = x22 + x22_res

        #####################################
        ###### stage3: (L, W) scale ########
        #####################################
        x33 = self.conv23(x22)
        x33 = F.relu(self.ps23(x33))
        #HyperTransformer at (L, W) scale
        x33_res = x33
        x33_res = torch.cat((x33_res, T_lv1), dim=1)
        x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
        S_lv1   = S_lv1.view(S_lv1.shape[0], S_lv1.shape[1], 1, 1)
        x33_res = x33_res * S_lv1.expand_as(x33_res)
        x33     = x33 + x33_res
        #Residual learning
        x33_res = x33
        for i in range(self.num_res_blocks[3]):
            x33_res = self.RB33[i](x33_res)
        x33_res = self.conv33_tail(x33_res)
        x33 = x33 + x33_res

        #####################################
        ############ Feature Pyramid ########
        #####################################
        x11_up  = F.interpolate(x11, scale_factor=4, mode='bicubic')
        x22_up  = F.interpolate(x22, scale_factor=2, mode='bicubic')
        xF      = torch.cat((x11_up, x22_up, x33), dim=1)
        
        #####################################
        ####  Final convolution   ###########
        #####################################
        xF      = self.final_conv(xF)
        xF_res  = xF

        #Final resblocks
        for i in range(self.num_res_blocks[4]):
            xF_res = self.RBF[i](xF_res)
        xF_res  = self.convF_tail(xF_res)
        x       = xF + xF_res

        #####################################
        #      Perceptual loss              #
        #####################################
        Phi_LFE_lv1, Phi_LFE_lv2, Phi_LFE_lv3 = self.LFE_HSI(x)
        #Transferal perceptual loss
        loss_tp = LOSS_TP(Phi_LFE_lv1, T_lv1)+LOSS_TP(Phi_LFE_lv2, T_lv2)+LOSS_TP(Phi_LFE_lv3, T_lv3)
        output = {  "pred": x, 
                    "tp_loss": loss_tp}
        return output


#######################################
# Hyperspectral Transformer (HSIT) ####
#######################################
class HyperTransformerBASIC(nn.Module):
    def __init__(self, config):
        super(HyperTransformerBASIC, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.factor         = config[config["train_dataset"]]["factor"]
        self.config         = config

        self.num_res_blocks = [16, 1, 1, 1, 4]
        self.n_feats        = 256
        self.res_scale      = 1

        self.LFE_HSI    = LFE(in_channels=self.in_channels)
        self.LFE_PAN    = LFE(in_channels=1)
        self.TS         = TS_Hard()
        self.SFE        = SFE(self.in_channels, self.num_res_blocks[0], self.n_feats, self.res_scale)

        ###############
        ### stage11 ###
        ###############
        self.conv11_head    = conv3x3(2*self.n_feats, self.n_feats)
        self.conv12         = conv3x3(self.n_feats, self.n_feats*2)
        self.ps12           = nn.PixelShuffle(2)
        #Residial blocks
        self.RB11           = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels=self.n_feats, out_channels=self.n_feats,
                res_scale=self.res_scale))
        self.conv11_tail = conv3x3(self.n_feats, self.n_feats)

        ###############
        ### stage22 ###
        ###############
        self.conv22_head    = conv3x3(2*int(self.n_feats/2), int(self.n_feats/2))
        self.conv23         = conv3x3(int(self.n_feats/2), self.n_feats)
        self.ps23           = nn.PixelShuffle(2)
        #Residual blocks
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB22.append(ResBlock(in_channels=int(self.n_feats/2), out_channels=int(self.n_feats/2), res_scale=self.res_scale))
        self.conv22_tail = conv3x3(int(self.n_feats/2), int(self.n_feats/2))

        ###############
        ### stage33 ###
        ###############
        self.conv33_head    = conv3x3(2*int(self.n_feats/4), int(self.n_feats/4))
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB33.append(ResBlock(in_channels=int(self.n_feats/4), out_channels=int(self.n_feats/4), res_scale=self.res_scale))
        self.conv33_tail = conv3x3(int(self.n_feats/4), int(self.n_feats/4))

        ##############
        ### FINAL ####
        ##############
        self.final_conv     = nn.Conv2d(in_channels=self.n_feats+int(self.n_feats/2)+int(self.n_feats/4), out_channels=self.out_channels, kernel_size=3, padding=1)
        self.RBF = nn.ModuleList()
        for i in range(self.num_res_blocks[4]):
            self.RBF.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
        self.convF_tail = conv3x3(self.out_channels, self.out_channels)



    def forward(self, X_MS, X_PAN):
        with torch.no_grad():
            if not self.is_DHP_MS:
                X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor,self.factor),mode ='bicubic')

            else:
                X_MS_UP = X_MS
            
            # Generating PAN, and PAN (UD) images
            X_PAN   = X_PAN.unsqueeze(dim=1)
            PAN_D   = F.interpolate(X_PAN, scale_factor=(1/self.factor, 1/self.factor), mode ='bilinear')
            PAN_UD  = F.interpolate(PAN_D, scale_factor=(self.factor, self.factor), mode ='bilinear')

        #Extracting T and S at multiple-scales
        #lv1 = LxW, lv2 = (L/2)x(w/2), lv3=(L/4, W/4)
        V_lv1, V_lv2, V_lv3 = self.LFE_PAN(X_PAN)
        #K_lv1, K_lv2, K_lv3 = self.LFE_PAN(PAN_UD)
        #Q_lv1, Q_lv2, Q_lv3 = self.LFE_HSI(X_MS_UP)

        #T_lv3, S_lv3  = self.TS(V_lv3, K_lv3, Q_lv3)
        #T_lv2, S_lv2  = self.TS(V_lv2, K_lv2, Q_lv2)
        #T_lv1, S_lv1  = self.TS(V_lv1, K_lv1, Q_lv1)

        #Save feature maps for illustration purpose
        # feature_dic={}
        # f = 1e3
        # feature_dic.update({"V": f*V_lv3.detach().cpu().numpy()})
        # feature_dic.update({"K": f*K_lv3.detach().cpu().numpy()})
        # feature_dic.update({"Q": f*Q_lv3.detach().cpu().numpy()})
        # feature_dic.update({"T": f*T_lv3.detach().cpu().numpy()})
        # savemat("/home/lidan/Dropbox/Hyperspectral/HyperTransformer/feature_visualization_pavia/features_lv3.mat", feature_dic)
        # exit()

        #Shallow Feature Extraction (SFE)
        x = self.SFE(X_MS)

        #####################################
        #### stage1: (L/4, W/4) scale ######
        #####################################
        x11 = x
        #HyperTransformer at (L/4, W/4) scale
        x11_res = x11
        x11_res = torch.cat((x11_res, V_lv3), dim=1)
        x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
        #S_lv3   = S_lv3.view(S_lv3.shape[0], S_lv3.shape[1], 1, 1)
        #x11_res = x11_res * S_lv3.expand_as(x11_res)
        x11     = x11 + x11_res
        #Residial learning
        x11_res = x11
        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res

        #####################################
        #### stage2: (L/2, W/2) scale ######
        #####################################
        x22 = self.conv12(x11)
        x22 = F.relu(self.ps12(x22))
        #HyperTransformer at (L/2, W/2) scale
        x22_res = x22
        x22_res = torch.cat((x22_res, V_lv2), dim=1)
        x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
        #S_lv2   = S_lv2.view(S_lv2.shape[0], S_lv2.shape[1], 1, 1)
        #x22_res = x22_res * S_lv2.expand_as(x22_res)
        x22     = x22 + x22_res
        #Residial learning
        x22_res = x22
        for i in range(self.num_res_blocks[2]):
            x22_res = self.RB22[i](x22_res)
        x22_res = self.conv22_tail(x22_res)
        x22 = x22 + x22_res

        #####################################
        ###### stage3: (L, W) scale ########
        #####################################
        x33 = self.conv23(x22)
        x33 = F.relu(self.ps23(x33))
        #HyperTransformer at (L, W) scale
        x33_res = x33
        x33_res = torch.cat((x33_res, V_lv1), dim=1)
        x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
        #S_lv1   = S_lv1.view(S_lv1.shape[0], S_lv1.shape[1], 1, 1)
        #x33_res = x33_res * S_lv1.expand_as(x33_res)
        x33     = x33 + x33_res
        #Residual learning
        x33_res = x33
        for i in range(self.num_res_blocks[3]):
            x33_res = self.RB33[i](x33_res)
        x33_res = self.conv33_tail(x33_res)
        x33 = x33 + x33_res

        #####################################
        ############ Feature Pyramid ########
        #####################################
        x11_up  = F.interpolate(x11, scale_factor=4, mode='bicubic')
        x22_up  = F.interpolate(x22, scale_factor=2, mode='bicubic')
        xF      = torch.cat((x11_up, x22_up, x33), dim=1)
        
        #####################################
        ####  Final convolution   ###########
        #####################################
        xF      = self.final_conv(xF)
        xF_res  = xF

        #Final resblocks
        for i in range(self.num_res_blocks[4]):
            xF_res = self.RBF[i](xF_res)
        xF_res  = self.convF_tail(xF_res)
        x       = xF + xF_res

        #####################################
        #      Perceptual loss              #
        #####################################
        #Phi_LFE_lv1, Phi_LFE_lv2, Phi_LFE_lv3 = self.LFE_HSI(x)
        #Transferal perceptual loss
        #if self.config[self.config["train_dataset"]]["Transfer_Periferal_Loss"]:
        #loss_tp = LOSS_TP(Phi_LFE_lv1, T_lv1)+LOSS_TP(Phi_LFE_lv2, T_lv2)+LOSS_TP(Phi_LFE_lv3, T_lv3)
        output = {  "pred": x }
        return output

#######################################
# Hyperspectral Transformer (HSIT) ####
#######################################
# Experimenting with soft attention
class HyperTransformerSOFT(nn.Module):
    def __init__(self, config):
        super(HyperTransformerSOFT, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.factor         = config[config["train_dataset"]]["factor"]
        self.config         = config

        self.num_res_blocks = [1, 1, 1, 1, 1]
        self.n_feats        = 256
        self.res_scale      = 1

        self.LFE_HSI    = LFE(in_channels=self.in_channels)
        self.LFE_PAN    = LFE(in_channels=1)
        self.TS         = TS_Soft()
        self.SFE        = SFE(self.in_channels, self.num_res_blocks[0], self.n_feats, self.res_scale)

        ###############
        ### stage11 ###
        ###############
        self.conv11_head    = conv3x3(2*self.n_feats, self.n_feats)
        self.conv12         = conv3x3(self.n_feats, self.n_feats*2)
        self.ps12           = nn.PixelShuffle(2)
        #Residial blocks
        self.RB11           = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels=self.n_feats, out_channels=self.n_feats,
                res_scale=self.res_scale))
        self.conv11_tail = conv3x3(self.n_feats, self.n_feats)

        ###############
        ### stage22 ###
        ###############
        self.conv22_head    = conv3x3(2*int(self.n_feats/2), int(self.n_feats/2))
        self.conv23         = conv3x3(int(self.n_feats/2), self.n_feats)
        self.ps23           = nn.PixelShuffle(2)
        #Residual blocks
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB22.append(ResBlock(in_channels=int(self.n_feats/2), out_channels=int(self.n_feats/2), res_scale=self.res_scale))
        self.conv22_tail = conv3x3(int(self.n_feats/2), int(self.n_feats/2))

        ###############
        ### stage33 ###
        ###############
        self.conv33_head    = conv3x3(2*int(self.n_feats/4), int(self.n_feats/4))
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB33.append(ResBlock(in_channels=int(self.n_feats/4), out_channels=int(self.n_feats/4), res_scale=self.res_scale))
        self.conv33_tail = conv3x3(int(self.n_feats/4), int(self.n_feats/4))

        ##############
        ### FINAL ####
        ##############
        self.final_conv     = nn.Conv2d(in_channels=self.n_feats+int(self.n_feats/2)+int(self.n_feats/4), out_channels=self.out_channels, kernel_size=3, padding=1)
        self.RBF = nn.ModuleList()
        for i in range(self.num_res_blocks[4]):
            self.RBF.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
        self.convF_tail = conv3x3(self.out_channels, self.out_channels)



    def forward(self, X_MS, X_PAN):
        with torch.no_grad():
            if not self.is_DHP_MS:
                X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor,self.factor),mode ='bicubic')

            else:
                X_MS_UP = X_MS
            
            # Generating PAN, and PAN (UD) images
            X_PAN   = X_PAN.unsqueeze(dim=1)
            PAN_D   = F.interpolate(X_PAN, scale_factor=(1/self.factor, 1/self.factor), mode ='bilinear')
            PAN_UD  = F.interpolate(PAN_D, scale_factor=(self.factor, self.factor), mode ='bilinear')

        #Extracting T and S at multiple-scales
        #lv1 = LxW, lv2 = (L/2)x(w/2), lv3=(L/4, W/4)
        V_lv1, V_lv2, V_lv3 = self.LFE_PAN(X_PAN)
        K_lv1, K_lv2, K_lv3 = self.LFE_PAN(PAN_UD)
        Q_lv1, Q_lv2, Q_lv3 = self.LFE_HSI(X_MS_UP)

        T_lv3, S_lv3  = self.TS(V_lv3, K_lv3, Q_lv3)
        T_lv2, S_lv2  = self.TS(V_lv2, K_lv2, Q_lv2)
        T_lv1, S_lv1  = self.TS(V_lv1, K_lv1, Q_lv1)

        #Save feature maps for illustration purpose
        # feature_dic={}
        # feature_dic.update({"V": V_lv1.detach().cpu().numpy()})
        # feature_dic.update({"K": K_lv1.detach().cpu().numpy()})
        # feature_dic.update({"Q": Q_lv1.detach().cpu().numpy()})
        # feature_dic.update({"T": T_lv1.detach().cpu().numpy()})
        # savemat("/home/lidan/Dropbox/Hyperspectral/HyperTransformer/feature_visualization_pavia/features_soft_lv2.mat", feature_dic)
        # exit()

        #Shallow Feature Extraction (SFE)
        x = self.SFE(X_MS)

        #####################################
        #### stage1: (L/4, W/4) scale ######
        #####################################
        x11 = x
        #HyperTransformer at (L/4, W/4) scale
        x11_res = x11
        x11_res = torch.cat((x11_res, T_lv3), dim=1)
        x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
        #S_lv3   = S_lv3.view(S_lv3.shape[0], S_lv3.shape[1], 1, 1)
        #x11_res = x11_res * S_lv3.expand_as(x11_res)
        x11     = x11 + x11_res
        #Residial learning
        x11_res = x11
        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res

        #####################################
        #### stage2: (L/2, W/2) scale ######
        #####################################
        x22 = self.conv12(x11)
        x22 = F.relu(self.ps12(x22))
        #HyperTransformer at (L/2, W/2) scale
        x22_res = x22
        x22_res = torch.cat((x22_res, T_lv2), dim=1)
        x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
        #S_lv2   = S_lv2.view(S_lv2.shape[0], S_lv2.shape[1], 1, 1)
        #x22_res = x22_res * S_lv2.expand_as(x22_res)
        x22     = x22 + x22_res
        #Residial learning
        x22_res = x22
        for i in range(self.num_res_blocks[2]):
            x22_res = self.RB22[i](x22_res)
        x22_res = self.conv22_tail(x22_res)
        x22 = x22 + x22_res

        #####################################
        ###### stage3: (L, W) scale ########
        #####################################
        x33 = self.conv23(x22)
        x33 = F.relu(self.ps23(x33))
        #HyperTransformer at (L, W) scale
        x33_res = x33
        x33_res = torch.cat((x33_res, T_lv1), dim=1)
        x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
        #S_lv1   = S_lv1.view(S_lv1.shape[0], S_lv1.shape[1], 1, 1)
        #x33_res = x33_res * S_lv1.expand_as(x33_res)
        x33     = x33 + x33_res
        #Residual learning
        x33_res = x33
        for i in range(self.num_res_blocks[3]):
            x33_res = self.RB33[i](x33_res)
        x33_res = self.conv33_tail(x33_res)
        x33 = x33 + x33_res

        #####################################
        ############ Feature Pyramid ########
        #####################################
        x11_up  = F.interpolate(x11, scale_factor=4, mode='bicubic')
        x22_up  = F.interpolate(x22, scale_factor=2, mode='bicubic')
        xF      = torch.cat((x11_up, x22_up, x33), dim=1)
        
        #####################################
        ####  Final convolution   ###########
        #####################################
        xF      = self.final_conv(xF)
        xF_res  = xF

        #Final resblocks
        for i in range(self.num_res_blocks[4]):
            xF_res = self.RBF[i](xF_res)
        xF_res  = self.convF_tail(xF_res)
        x       = xF + xF_res

        #####################################
        #      Perceptual loss              #
        #####################################
        Phi_LFE_lv1, Phi_LFE_lv2, Phi_LFE_lv3 = self.LFE_HSI(x)
        #Transferal perceptual loss
        loss_tp = LOSS_TP(Phi_LFE_lv1, T_lv1)+LOSS_TP(Phi_LFE_lv2, T_lv2)+LOSS_TP(Phi_LFE_lv3, T_lv3)

        #####################################
        #        VGG loss for Extractors    #
        #####################################
        Q_RGB   = torch.cat((torch.mean(outputs[:, 0:config[config["train_dataset"]]["G"], :, :], 1).unsqueeze(1), 
                                        torch.mean(outputs[:, config[config["train_dataset"]]["B"]:config[config["train_dataset"]]["R"], :, :], 1).unsqueeze(1), 
                                        torch.mean(outputs[:, config[config["train_dataset"]]["G"]:config[config["train_dataset"]]["spectral_bands"], :, :], 1).unsqueeze(1)), 1)
        output = {  "pred": x, 
                    "tp_loss": loss_tp}
        return output
