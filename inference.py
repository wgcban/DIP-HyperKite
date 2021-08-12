import os
import argparse
import json
import torch
import numpy as np
from torch.nn.functional import threshold, unfold
from dataloaders.HSI_datasets import *
from utils.logger import Logger
import torch.utils.data as data
from utils.helpers import initialize_weights, initialize_weights_new, to_variable, make_patches
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from models.models import MODELS
from utils.metrics import *
import shutil
import torchvision
from torch.distributions.uniform import Uniform
import sys
import kornia
from kornia import laplacian, sobel

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

__dataset__ = {"cave_dataset": cave_dataset, "pavia_dataset": pavia_dataset, "botswana_dataset": botswana_dataset, "paviaU_dataset": paviaU_dataset}

# PARSE THE ARGS
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-c', '--config', default='configs/config.json',type=str,
                        help='Path to the config file')
parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
parser.add_argument('--local', action='store_true', default=False)
args = parser.parse_args()

# LOADING THE CONFIG FILE
config = json.load(open(args.config))
torch.backends.cudnn.benchmark = True

# SEEDS
torch.manual_seed(7)

# NUMBER OF GPUs
num_gpus = torch.cuda.device_count()

# MODEL
model = MODELS[config["model"]](config)
print(f'\n{model}\n')

# SENDING MODEL TO DEVICE
if num_gpus > 1:
    print("Training with multiple GPUs ({})".format(num_gpus))
    model = nn.DataParallel(model).cuda()
else:
    print("Single Cuda Node is avaiable")
    model.cuda()

# DATA LOADERS
print("Training with dataset => {}".format(config["train_dataset"]))
train_loader = data.DataLoader(
                        __dataset__[config["train_dataset"]](
                            config,
                            is_train=True,
                            want_DHP_MS_HR=config["is_DHP_MS"],
                        ),
                        batch_size=config["train_batch_size"],
                        num_workers=config["num_workers"],
                        shuffle=True,
                        pin_memory=False,
                    )

test_loader = data.DataLoader(
                        __dataset__[config["train_dataset"]](
                            config,
                            is_train=False,
                            want_DHP_MS_HR=config["is_DHP_MS"],
                        ),
                        batch_size=config["val_batch_size"],
                        num_workers=config["num_workers"],
                        shuffle=True,
                        pin_memory=False,
                    )

# INITIALIZATION OF PARAMETERS
start_epoch = 1
total_epochs = config["trainer"]["total_epochs"]

# OPTIMIZER
if config["optimizer"]["type"] == "SGD":
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config["optimizer"]["args"]["lr"], 
        momentum = config["optimizer"]["args"]["momentum"], 
        weight_decay= config["optimizer"]["args"]["weight_decay"]
    )
elif config["optimizer"]["type"] == "ADAM":
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["optimizer"]["args"]["lr"],
        weight_decay= config["optimizer"]["args"]["weight_decay"]
    )
else:
    exit("Undefined optimizer type")

# LEARNING RATE SHEDULER
scheduler = optim.lr_scheduler.StepLR(  optimizer, 
                                        step_size=config["optimizer"]["step_size"], 
                                        gamma=config["optimizer"]["gamma"])

# IF RESUME
if args.resume is not None:
    print("Loading from existing FCN and copying weights to continue....")
    print("This part not implemented yet")
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint, strict=True)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    # initialize_weights(model)
    initialize_weights_new(model)

# LOSS
if config["loss_type"] == "L1":
    criterion   = torch.nn.L1Loss()
    HF_loss     = torch.nn.L1Loss()
elif config["loss_type"] == "MSE":
    criterion   = torch.nn.MSELoss()
    HF_loss     = torch.nn.MSELoss()
else:
    exit("Undefined loss type")
    
# TEST EPPOCH
def test(epoch):
    test_loss   = 0.0
    cc          = 0.0
    sam         = 0.0
    rmse        = 0.0
    ergas       = 0.0
    psnr        = 0.0
    val_outputs = {}
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            _, MS_image, PAN_image, reference = data

            # Generating small patches
            if config["trainer"]["is_small_patch_train"]:
                MS_image, unfold_shape = make_patches(MS_image, patch_size=config["trainer"]["patch_size"])
                PAN_image, _ = make_patches(PAN_image, patch_size=config["trainer"]["patch_size"])
                reference, _ = make_patches(reference, patch_size=config["trainer"]["patch_size"])

            # Inputs and references...
            MS_image = MS_image.float().cuda()
            PAN_image =PAN_image.float().cuda()
            reference = reference.float().cuda()

            # Taking model output
            outputs = model(MS_image, PAN_image)

            # Computing validation loss
            loss = criterion(outputs, reference)
            test_loss += loss.item()

            # Quantization of output vectors
            # Normalizing to zero mean and unit variance
            ms_mean     = torch.FloatTensor(config[config["train_dataset"]]["ms_mean"]).cuda()
            ms_std      = torch.FloatTensor(config[config["train_dataset"]]["ms_std"]).cuda()
            outputs     = (outputs.permute(0,2,3,1)*ms_std + ms_mean).permute(0,3, 1, 2)
            reference   = (reference.permute(0,2,3,1)*ms_std + ms_mean).permute(0,3, 1, 2)
            MS_image    = (MS_image.permute(0,2,3,1)*ms_std + ms_mean).permute(0,3, 1, 2)
            outputs[outputs>1.0] = 1.0
            outputs[outputs<0.0] = 0.0
            outputs = torch.round(outputs*255)/255.0
            #reference = torch.round(reference)/255.0
            
            ### Computing performance metrics ###
            # Cross-correlation
            cc += cross_correlation(outputs, reference)
            # SAM
            sam += SAM(outputs, reference)
            # RMSE
            rmse += RMSE(outputs, reference)
            # ERGAS
            beta = torch.tensor(config[config["train_dataset"]]["HR_size"]/config[config["train_dataset"]]["LR_size"]).cuda()
            ergas += ERGAS(outputs, reference, beta)
            # PSNR
            psnr += PSNR(outputs, reference)

    # Taking average of performance metrics over test set
    cc /= len(test_loader)
    sam /= len(test_loader)
    rmse /= len(test_loader)
    ergas /= len(test_loader)
    psnr /= len(test_loader)

    # Writing test results to tensorboard
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Test_Metrics/CC', cc, epoch)
    writer.add_scalar('Test_Metrics/SAM', sam, epoch)
    writer.add_scalar('Test_Metrics/RMSE', rmse, epoch)
    writer.add_scalar('Test_Metrics/ERGAS', ergas, epoch)
    writer.add_scalar('Test_Metrics/PSNR', psnr, epoch)

    # Images to tensorboard
    # Regenerating the final image
    if config["trainer"]["is_small_patch_train"]:
        outputs = outputs.view(unfold_shape).permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        outputs = outputs.contiguous().view(config["val_batch_size"], 
                                            config[config["train_dataset"]]["spectral_bands"],
                                            config[config["train_dataset"]]["HR_size"],
                                            config[config["train_dataset"]]["HR_size"])
        reference = reference.view(unfold_shape).permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        reference = reference.contiguous().view(config["val_batch_size"], 
                                                config[config["train_dataset"]]["spectral_bands"],
                                                config[config["train_dataset"]]["HR_size"],
                                                config[config["train_dataset"]]["HR_size"])
        MS_image = MS_image.view(unfold_shape).permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        MS_image = MS_image.contiguous().view(config["val_batch_size"], 
                                                config[config["train_dataset"]]["spectral_bands"],
                                                config[config["train_dataset"]]["HR_size"],
                                                config[config["train_dataset"]]["HR_size"])
    ms      = torch.unsqueeze(MS_image.view(-1, MS_image.shape[-2], MS_image.shape[-1]), 1)
    pred    = torch.unsqueeze(outputs.view(-1, outputs.shape[-2], outputs.shape[-1]), 1)
    ref     = torch.unsqueeze(reference.view(-1, reference.shape[-2], reference.shape[-1]), 1)
    imgs    = torch.zeros(5*pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3])
    for i in range(pred.shape[0]):
        imgs[5*i]   = ms[i]
        imgs[5*i+1] = torch.abs(ms[i]-pred[i])/torch.max(torch.abs(ms[i]-pred[i]))
        imgs[5*i+2] = pred[i]
        imgs[5*i+3] = ref[i]
        imgs[5*i+4] = torch.abs(ref[i]-ms[i])/torch.max(torch.abs(ref[i]-ms[i]))
    imgs = torchvision.utils.make_grid(imgs, nrow=5)
    writer.add_image('Images', imgs, epoch)

    #Return Outputs
    outputs = { "loss": float(test_loss), 
                "cc": float(cc), 
                "sam": float(sam), 
                "rmse": float(rmse), 
                "ergas": float(ergas), 
                "psnr": float(psnr)}
    return outputs

# SETTING UP TENSORBOARD and COPY JSON FILE TO SAVE DIRECTORY
PATH = "./"+config["experim_name"]+"/"+config["train_dataset"]+"inference"+"/"+"N_modules("+str(config["N_modules"])+")"
ensure_dir(PATH+"/")
writer = SummaryWriter(log_dir=PATH)
shutil.copy2(args.config, PATH)

# Print model to text file
original_stdout = sys.stdout 
with open(PATH+"/"+"model_summary.txt", 'w+') as f:
    sys.stdout = f
    print(f'\n{model}\n')
    sys.stdout = original_stdout 

# MAIN LOOP
best_psnr=0.0
for epoch in range(1):
    print("\nStart Inference: %d" % epoch)

    outputs=test(epoch)
        


    