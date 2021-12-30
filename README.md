[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hyperspectral-pansharpening-based-on-improved/super-resolution-on-botswana)](https://paperswithcode.com/sota/super-resolution-on-botswana?p=hyperspectral-pansharpening-based-on-improved)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hyperspectral-pansharpening-based-on-improved/super-resolution-on-pavia-centre)](https://paperswithcode.com/sota/super-resolution-on-pavia-centre?p=hyperspectral-pansharpening-based-on-improved)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hyperspectral-pansharpening-based-on-improved/image-super-resolution-on-chikusei-dataset)](https://paperswithcode.com/sota/image-super-resolution-on-chikusei-dataset?p=hyperspectral-pansharpening-based-on-improved)

# DIP-HyperKite: Hyperspectral Pansharpening Based on Improved Deep Image Prior and  Residual Reconstruction
[Wele Gedara Chaminda Bandara](https://www.researchgate.net/profile/Chaminda-Bandara-4), [Jeya Maria Jose Valanarasu](https://jeya-maria-jose.github.io/research/), [Vishal M. Patel](https://engineering.jhu.edu/vpatel36/sciencex_teams/vishalpatel/)

Accepted for Publication in IEEE Transactions on Geoscience and Remote Sensing. Download paper: [IEEE-Xplore](https://ieeexplore.ieee.org/document/9664535) or [arXiv](https://arxiv.org/abs/2107.02630).

## Abstract
Hyperspectral pansharpening aims to synthesize a low-resolution hyperspectral image (LR-HSI) with a registered panchromatic image (PAN) to generate an enhanced HSI with high spectral and spatial resolution.  Recently proposed HS pansharpening methods have obtained remarkable results using deep convolutional networks (ConvNets), which typically consist of three steps: (1) up-sampling the LR-HSI, (2) predicting the residual image via a ConvNet, and (3) obtaining the final fused HSI by adding the outputs from first and second steps.  Recent methods have leveraged Deep Image Prior (DIP) to up-sample the LR-HSI due to its excellent ability to preserve both spatial and spectral information, without learning from  large data sets. However, we observed that the quality of up-sampled HSIs can be further improved by introducing an additional spatial-domain constraint to the conventional spectral-domain energy function. We define our spatial-domain constraint as the $L_1$ distance between the predicted PAN image and the actual PAN image. To estimate the PAN image of the up-sampled HSI, we also propose a learnable spectral response function (SRF). Moreover, we noticed that the residual image between the up-sampled HSI and the reference HSI mainly consists of edge information and very fine structures. In order to accurately estimate fine information, we propose a novel over-complete network, called HyperKite, which focuses on learning high-level features by constraining the receptive from increasing in the deep layers.

## Citation
    @inproceedings{bandara2021hyperspectral,
    title={Hyperspectral Pansharpening Based on Improved Deep Image Prior and Residual Reconstruction},
    author={Bandara, Wele Gedara Chaminda and Valanarasu, Jeya Maria Jose and Patel, Vishal M},
    journal={arXiv preprint arXiv:2107.02630},
    year={2021}
    } 

## DIP-HyperKite Network
<p align="center">
<img src="imgs/GRS-method.jpg"/>

## Proposed Spatial Loss Function for DIP Process
<p align="center">
<img src="imgs/GRS-R.jpg" width="600px"/>

## Proposed HyperKite Network for Residual Prediction
<p align="center">
<img src="imgs/GRS-HyperKite.jpg"/>

## Downloading Datasets
In this paper, we used three publically available datasets and the link to each dataset is given below:
1. Pavia Center Dataset: [Click Here](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
2. Botswana Dataset: [Click Here](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
3. Chikusei Dataset: [Click Here](https://naotoyokoya.com/Download.html)

Once you downloaded the dataset in `.mat` format save them in respective folders: `./datasets/pavia_centre/`, `./datasets/botswana/`, and `./datasets/chikusei/`.
## Install Requirements
The `requirements.txt` file lists all Python libraries required for DIP-HyperKite. To install requirements first create a new conda environment and then install all required libraries using following commands.

`conda create --name hyperkite`

`conda activate hyperkite`

`conda install --file requirements.txt`

## Generating LR-HSIs, PAN images, and Ref HSIs
Next, we generate LR-HSIs, PAN images, and Ref-HSIs required to train the pansharpening model using the famous Wald's protocol. For this, you simply needs to run the `process_pavia.m`, `process_botswana.m`, and `process_chikusei.m` files in the `./datasets/pavia_centre/`, `./datasets/botswana/`, and `./datasets/chikusei/`, respectively.

## Upsampling using DIP
To generate the up-sampled version of LR-HSIs, you need to run the following code. 
Please make sure to change the first few lines of the `./configs/config_dhp.json` as you want. 
Basically you want to change the `experiment_name` and `dataset` you want to run.
    `CUDA_VISIBLE_DEVICES=0 python train_dhp.py --config ./configs/config_dhp.json`

You need to repreat this for all the three datasets.

## Training the HyperKite
Once you generated the DIP upsampled HSIs, now you are ready to train the HyperKite network. You can train the HyperKite network by executing the following command.
    `CUDA_VISIBLE_DEVICES=0 python train.py --config ./configs/config.json`

## Visualing the Results via Tensorboard
All the results will be saved in `./Experiments/` folder. You can visualize all the performance metrics by executing the following command.
    `tensorboard --logdir ./Experiments/Vxx/pavia_center/`
    
## Qaulitative Results on [Pavia Center Dataset](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
<p align="center">
<img src="imgs/GRS2-final_pred_pavia.jpg"/>

## Qaulitative Results on [Botswana Dataset](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
<p align="center">
<img src="imgs/GRS2-final_pred_botswana.jpg"/>

## Qaulitative Results on [Chikusei Dataset](https://naotoyokoya.com/Download.html)
<p align="center">
<img src="imgs/GRS2-final_pred_chikusei.jpg"/>
    
If you find our paper useful, please cite our paper:
`
    @ARTICLE{9664535,  author={Bandara, Wele Gedara Chaminda and Valanarasu, Jeya Maria Jose and Patel, Vishal M.},  journal={IEEE Transactions on Geoscience and Remote Sensing},   title={Hyperspectral Pansharpening Based on Improved Deep Image Prior and Residual Reconstruction},   year={2021},  volume={},  number={},  pages={1-1},  doi={10.1109/TGRS.2021.3139292}}
    
`
