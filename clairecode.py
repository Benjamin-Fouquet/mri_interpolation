from os.path import expanduser
import sys
from tqdm import tqdm
import os

from torchio.data.image import ScalarImage
home = expanduser("~")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import monai
import matplotlib.pyplot as plt
import math
from skimage.metrics import structural_similarity as ssim

import argparse

from models.DRIT import DRIT
from models.DenseNet import Dense, DenseNet
from models.UNet_2D import UNet
from models.HighResNet import HighResNet

def compute_PSNR(max_val,prediction: torch.Tensor,ground_truth: torch.Tensor):
    prediction=prediction.float()
    ground_truth=ground_truth.float()
    mse=torch.nn.functional.mse_loss(prediction,ground_truth)
    if mse==0:
        return(100)
    else:
        return(20 * math.log10(max_val) - 10 * torch.log10(mse))

def compute_PSNR_onMask(max_val,prediction: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor):
    prediction=prediction.float()
    ground_truth=ground_truth.float()
    mask=mask.float()
    MSE=((prediction-ground_truth)**2)
    MSE=MSE*mask
    Nombre_pixels_masque=mask.sum()
    MSE=MSE/(Nombre_pixels_masque)
    MSE=MSE.sum()
    return(20 * math.log10(max_val) - 10 * torch.log10(MSE))

def compute_PSNR_outOfMask(max_val,prediction: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor):
    prediction=prediction.float()
    ground_truth=ground_truth.float()
    mask=mask.float()
    mask=torch.abs((mask-1)*(-1))
    MSE=((prediction-ground_truth)**2)
    MSE=MSE*mask
    Nombre_pixels_masque=mask.sum()
    MSE=MSE/(Nombre_pixels_masque)
    MSE=MSE.sum()
    return(20 * math.log10(max_val) - 10 * torch.log10(MSE))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo TorchIO inference')
    parser.add_argument('-i', '--input', help='Input image', type=str, required=True)
    parser.add_argument('-m', '--model', help='Pytorch model', type=str, required=True)
    parser.add_argument('-o', '--output', help='Output image', type=str, required=True)
    parser.add_argument('-F', '--fuzzy', help='Output fuzzy image', type=str, required=False)
    parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = (64,64,1))
    parser.add_argument('--patch_overlap', help='Patch overlap', type=int, required=False, default = (16,16,0))
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 2)
    parser.add_argument('-g', '--ground_truth', help='Ground truth for metric computation', type=str, required=False)
    parser.add_argument('-T', '--test_time', help='Number of inferences for test-time augmentation', type=int, required=False, default=1)
    parser.add_argument('-c', '--channels', help='Number of channels', type=int, required=False, default=16)
    parser.add_argument('-f', '--features', help='Number of features', type=int, required=False, default=32)
    parser.add_argument('--classes', help='Number of classes', type=int, required=False, default=1)
    parser.add_argument('-s', '--scales', help='Scaling factor (test-time augmentation)', type=float, required=False, default=0.05)
    parser.add_argument('-d', '--degrees', help='Rotation degrees (test-time augmentation)', type=int, required=False, default=10)
    parser.add_argument('-D', '--dataset', help='Name of the dataset used', type=str, required=False, default='hcp')
    parser.add_argument('-t', '--test_image', help='Image test (skip inference and goes directly to PSNR)', type=str, required=False)
    parser.add_argument('-S', '--segmentation', help='Segmentation to use', type=str,required=False)
    parser.add_argument('-n', '--network', help='Network to use (UNet, ResNet, disentangled..)', type=str,required=True)
    parser.add_argument('--gpu', help='GPU to use (If None, goes on CPU)', type=str,required=False)

    args = parser.parse_args()

    dataset=args.dataset
    model=args.model
    network=args.network
    gpu=args.gpu
    if gpu is not None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:"+gpu if use_cuda else "cpu")


    #############################################################################################################################################################################""
    if network=='Disentangled':
        Net= DRIT(
            criterion=nn.MSELoss(),
            dataset='equinus',     
            learning_rate=1e-4,
            optimizer_class=torch.optim.Adam,
            #optimizer_class=torch.optim.Adam,
            n_features = args.features,
            prefix = '',
        )
    elif network=='UNet':
        Net= UNet(
            criterion=nn.MSELoss(),
            dataset='equinus',     
            learning_rate=1e-4,
            optimizer_class=torch.optim.Adam,
            #optimizer_class=torch.optim.Adam,
            n_features = args.features,
            prefix = '',
        )
    elif network=='DenseNet':
        Net= Dense(
            criterion=nn.MSELoss(),
            dataset='equinus',     
            learning_rate=1e-4,
            optimizer_class=torch.optim.Adam,
            #optimizer_class=torch.optim.Adam,
            n_features = args.features,
            prefix = '',
        )
    elif network=='HighResNet':
        Net= HighResNet(
            criterion=nn.MSELoss(),
            dataset='equinus',     
            learning_rate=1e-4,
            optimizer_class=torch.optim.Adam,
            #optimizer_class=torch.optim.Adam,
            n_features = args.features,
            prefix = '',
        )
    else:
        sys.exit('Enter a valid network name')
    
    if model.split('/')[-1].split('.')[-1]=='pt':
        Net.load_state_dict(torch.load(model))
    else:
        Net.load_state_dict(torch.load(model)['state_dict'])
    Net.eval()
    if gpu is not None:
        Net.to(device=device)

    #############################################################################################################################################################################""
    
    I=args.input
    if args.segmentation==None:
        S=I.split('/')[:-1]
        s=I.split('/')[-1]
        ss=s.split('_')
        ss[0]='segmentation'
        ss=('_').join(ss)
        S.append(ss)
        S=('/').join(S)
    else:
        S=args.segmentation
    T2=args.ground_truth
    print(I)
    print(T2)
    # if os.path.exists(S):
    #     print(S)
    p='/home/aorus-users/claire/'
    t1=I.split('/')[-1]
    t1=p+t1
    s=S.split('/')[-1]
    s=p+s
    if args.ground_truth is not None:
        t2=T2.split('/')[-1]
        t2=p+t2


    subject = tio.Subject(
        T1_image=tio.ScalarImage(I),
        T2_image=tio.ScalarImage(T2),
        label=tio.LabelMap(S),
        )

    print(subject['T1_image'][tio.DATA].shape)
    print(subject['T2_image'][tio.DATA].shape)
    print(subject['label'][tio.DATA].shape)
    normalization = tio.ZNormalization(masking_method='label')
    #normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    # onehot = tio.OneHot()
    # spatial = tio.RandomAffine(scales=args.scales,degrees=args.degrees,translation=5,image_interpolation='bspline',p=1)
    # flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)

    #############################################################################################################################################################################""
    if args.test_image is None:
        print('Inference')
        batch_size = args.batch_size

        if I.split('/')[1]=='home':
            print('SUJET DÉJÀ NORMALISÉ')
            sub = subject
        else:
            print('NORMALISATION')
            augment = normalization
            sub = augment(subject)

        if model.find('(64, 64, 1)')!=-1:
            patch_size=(64,64,1)
            patch_overlap=(16,16,0)
        elif model.find('(1, 64, 64)')!=-1:
            patch_size=(1,64,64)
            patch_overlap=(0,16,16)
        else:
            sys.exit("Jeter un coup d'oeil a la taille de patchs")

        print('Patch size: '+str(patch_size))
        print('Patch overlap: '+str(patch_overlap))

        grid_sampler = tio.inference.GridSampler(
            sub,
            patch_size,
            patch_overlap
            )

        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
        aggregator = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

        if model.find('(64, 64, 1)')!=-1:
            with torch.no_grad():
                for patches_batch in tqdm(patch_loader):
                    input_tensor = patches_batch['T1_image'][tio.DATA]
                    if gpu is not None:
                        input_tensor=input_tensor.to(device)
                    input_tensor=input_tensor.squeeze(-1)
                    locations = patches_batch[tio.LOCATION]
                    if gpu is not None:
                        outputs = Net(input_tensor).cpu().detach()
                    else:
                        outputs = Net(input_tensor)
                    outputs=outputs.unsqueeze(-1)
                    aggregator.add_batch(outputs, locations)
        elif model.find('(1, 64, 64)')!=-1:
            with torch.no_grad():
                for patches_batch in tqdm(patch_loader):
                    input_tensor = patches_batch['T1_image'][tio.DATA]
                    locations = patches_batch[tio.LOCATION]
                    outputs = Net(input_tensor)
                    aggregator.add_batch(outputs, locations)
        else:
            sys.exit("Jeter un coup d'oeil a la taille de patchs (agregator)")

        output_tensor = aggregator.get_output_tensor()

        tmp = tio.ScalarImage(tensor=output_tensor, affine=sub.T1_image.affine)
        sub.add_image(tmp, 'T2_image_estim')

    #############################################################################################################################################################################""
    print('Saving images')
    if args.test_image is None:
        output_seg = tio.ScalarImage(tensor=output_tensor, affine=subject['T1_image'].affine)
        output_seg.save(args.output)
        

    if args.ground_truth is not None:
        gt_image= sub['T2_image']
        gt_image.save(t2)
        T1=sub['T1_image']
        T1.save(t1)
        label=sub['label']
        label.save(s)
        if args.test_image is None:
            pred_image = output_seg
        else:
            pred_image=tio.ScalarImage(args.test_image)