import os
import argparse
import json
from PIL.Image import NEAREST
import torch
import numpy as np
from torch._C import Value
from dataloaders.HSI_datasets import *
from utils.logger import Logger
import torch.utils.data as data
from utils.helpers import initialize_weights, to_variable
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import json
import cv2
#from models.models import MODELS
from models_dhp import *
from models_dhp.downsampler import Downsampler
from models_dhp.predict_PAN import Predict_PAN
from utils.metrics import *
from utils.sr_utils import *
import shutil
import torchvision
from scipy.io import savemat

dtype = torch.cuda.FloatTensor

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

__dataset__ = {"pavia_dataset": pavia_dataset, "botswana_dataset": botswana_dataset, "chikusei_dataset": chikusei_dataset, "botswana4_dataset": botswana4_dataset}

# PARSE THE ARGS
parser = argparse.ArgumentParser(description='===== PyTorch DHP Training =====')
parser.add_argument('-c', '--config', default='configs/config_dhp.json',type=str,
                        help='Path to the config file')
args = parser.parse_args()

# LOADING THE CONFIG FILE
config = json.load(open(args.config))
torch.backends.cudnn.benchmark = True

# SEEDS
torch.manual_seed(42)

# NUMBER OF GPUs
num_gpus = torch.cuda.device_count()

###### Configure parameres from .json file ######
input_depth     = config["model"]["input_depth"]
NET_TYPE        = config["model"]["NET_TYPE"] # UNet, ResNet
INPUT           = config["model"]["INPUT"]
pad             = config["model"]["pad"]
OPT_OVER        = config["model"]["OPT_OVER"]
KERNEL_TYPE     = config["model"]["KERNEL_TYPE"]
reg_noise_std   = config["model"]["reg_noise_std"]
factor          = config[config["train_dataset"]]["factor"]
LR              = config["optimizer"]["args"]["lr"]
OPTIMIZER       = config["optimizer"]["type"]
spectral_bands  = config[config["train_dataset"]]["spectral_bands"]
step_size       = config["optimizer"]["step_size"]
gamma           = config["optimizer"]["gamma"]


num_iter = config["trainer"]["total_epochs"]
reg_noise_std = 0.03


# INITIALIZATION OF PARAMETERS
start_epoch = 1

# TRAIN AND VALIDATION DATALOADERS
train_loader = data.DataLoader(
                        __dataset__[config["train_dataset"]](
                            config,
                            is_train=True,
                            is_dhp=True,
                        ),
                        batch_size=1,
                        num_workers=config["num_workers"],
                        shuffle=False,
                        pin_memory=False,
                    )


# SETTING UP TENSORBOARD and COPY JSON FILE TO SAVE DIRECTORY
ensure_dir("./"+config["experim_name"]+"/"+config["train_dataset"]+"/")
writer = SummaryWriter(log_dir=config["experim_name"]+"/"+config["train_dataset"])
shutil.copy2(args.config, "./"+config["experim_name"]+"/"+config["train_dataset"])
best_metrics = {}

# DEEP HYPERSPECTRAL PRIOR OVER TRAINING DATA
for i, data in enumerate(train_loader, 0):
    image_dict, MS_image, PAN_image, reference = data 
    
    # Get current image (folder) name
    img_name = image_dict["imgs"][0].split("/")[-1]     
    print("= Performing Deep Hyperspectral Prior on =>>> ["+img_name+"] dataset =")

    # Generate input noise tensor
    net_input = get_noise(input_depth, INPUT, (reference.shape[3], reference.shape[2])).type(dtype).detach()

    # Get model
    net = get_net(input_depth, n_channels=spectral_bands, 
            NET_TYPE        = NET_TYPE, 
            pad             = pad,
            skip_n33d       = 128, 
            skip_n33u       = 128, 
            skip_n11        = 4, 
            num_scales      = 5,
            upsample_mode   = 'bilinear').type(dtype)
    
    PAN_pred = Predict_PAN(spectral_bands=spectral_bands).type(dtype)

    # Loss
    if config["loss_type"] == "L1":
        loss_spectral   = torch.nn.L1Loss().type(dtype)
        loss_spatial    = torch.nn.L1Loss().type(dtype)
    elif config["loss_type"] == "MSE":
        loss_spectral   = torch.nn.MSELoss().type(dtype)
        loss_spatial    = torch.nn.MSELoss().type(dtype)
    else:
        exit("Undefined loss function.")

    # Convert LR image to variable
    img_LR_var = MS_image.type(dtype)
    reference = reference.type(dtype)
    PAN_image = PAN_image.type(dtype)

    # Setup downsampler
    downsampler = Downsampler(  n_planes    =   spectral_bands, 
                                factor      =   factor, 
                                kernel_type =   KERNEL_TYPE, 
                                phase       =   0.5, 
                                preserve_size=  True).type(dtype)

    # Setup closure
    psnr_best = 0.0

    def closure():
        global i, net_input, img_name, best_metrics, psnr_best, spectral_bands
        
        if reg_noise_std > 0:
            net_input               = net_input_saved + (noise.normal_() * reg_noise_std)
            #net_input[:, 31, :, :]  = PAN_image
        
        out_HR                  = net(net_input)
        predicted_PAN           = PAN_pred(net_output=out_HR, mode=config["spatial_avg_method"])
        out_LR                  = downsampler(out_HR)
        total_loss              = loss_spectral(out_LR, img_LR_var)

        if config["spatial_loss"]:
            alpha = config["alpha"]
            total_loss += alpha*loss_spatial(predicted_PAN, PAN_image)
            
        total_loss.backward()

        with torch.no_grad():
            pred            = out_HR.detach()
            pred[pred<0.0]  = 0.0
            pred[pred>1.0]  = 1.0 
            pred            = torch.round(pred*config[config["train_dataset"]]["max_value"])
            dhp_dic         = {"dhp": torch.squeeze(pred).permute(1,2,0).cpu().numpy()}
            ref             = torch.round(reference.detach()*config[config["train_dataset"]]["max_value"])

        # Computing performance metrics and wrting to tensorboard
        cc      = cross_correlation(pred, ref)
        sam     = SAM(pred, ref)
        rmse    = RMSE(pred/config[config["train_dataset"]]["max_value"], ref/config[config["train_dataset"]]["max_value"])
        beta    = torch.tensor(config[config["train_dataset"]]["HR_size"]/config[config["train_dataset"]]["LR_size"]).cuda()
        ergas   = ERGAS(pred, ref, beta)
        psnr    = PSNR(pred, ref)

        # Writing metrics to tensorboard
        writer.add_scalar('Metrics/'+img_name+'/CC', cc, i)
        writer.add_scalar('Metrics/'+img_name+'/SAM', sam, i)
        writer.add_scalar('Metrics/'+img_name+'/RMSE', rmse, i)
        writer.add_scalar('Metrics/'+img_name+'/ERGAS', ergas, i)
        writer.add_scalar('Metrics/'+img_name+'/PSNR', psnr, i)

        if (psnr > psnr_best) and (i>1000):
            # Update best psnr
            psnr_best = psnr

            # Save best metrics to .json file
            dict = {img_name: {"cc": cc.item(), "sam": sam.item(), "rmse": rmse.item(), "ergas": ergas.item(), "psnr": psnr.item()}}
            best_metrics.update(dict)
            with open("./"+config["experim_name"]+ "/" + config["train_dataset"] + "/"+"best_metrics.json", "w+") as outfile: 
                json.dump(best_metrics, outfile)

            # Scaling 
            pred        = pred/config[config["train_dataset"]]["max_value"]
            ref         = ref/config[config["train_dataset"]]["max_value"]

            #Write to tensorboard ....
            pred    = torch.unsqueeze(pred.detach().view(-1, pred.shape[-2], pred.shape[-1]), 1)
            ref     = torch.unsqueeze(ref.view(-1, ref.shape[-2], ref.shape[-1]), 1)

            imgs    = torch.zeros(2*pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3])
            for img_idx in range(pred.shape[0]):
                imgs[2*img_idx]     = ref[img_idx]
                imgs[2*img_idx+1]   = pred[img_idx]
            imgs = torchvision.utils.make_grid(imgs, nrow=2)
            #writer.add_image('Images/'+img_name+'/', imgs, i)
            writer.add_image('Images/'+img_name+'/', imgs)

            # Save upsampled images obtained from DHP
            savemat(image_dict["imgs"][0][:-4]+"_dhp_"+"{0:0=1d}".format(int(10*alpha))+ ".mat", dhp_dic)
            #savemat(image_dict["imgs"][0][:-4]+"_dhp_spectral.mat", dhp_dic)

        i += 1
        return total_loss


    net_input_saved = net_input.detach().clone()
    noise           = net_input.detach().clone()

    i = 0
    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter, step_size, gamma)


