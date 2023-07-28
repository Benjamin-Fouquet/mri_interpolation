'''
Temp script for interpolation tests, shoudl be deleted

scipy interp too long for correct results
'''
from scipy.interpolate import griddata, interpn, LinearNDInterpolator
from typing import List, Optional, Union
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
import tinycudann as tcnn 
import torch
import pytorch_lightning as pl 
import torch.nn.functional as F
import nibabel as nib 
from dataclasses import dataclass
import os
from types import MappingProxyType
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from math import pi
import torch.utils.data
import matplotlib.pyplot as plt
import encoding
from skimage import metrics
import itk

torch.manual_seed(1337)

@dataclass
class BaseConfig:
    checkpoint_path = None #'lightning_logs/version_384/checkpoints/epoch=99-step=100.ckpt'
    # image_path: str = '/mnt/Data/FetalAtlas/template_T2.nii.gz'
    image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_shape = nib.load(image_path).shape
    batch_size: int = 10000 #~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 50
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None 
    encoder_type: str = 'hash' #   
    # Network parameters
    n_levels: int = 8
    n_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: MappingProxyType = (64, 64, 8)
    finest_resolution: MappingProxyType = (512, 512, 8)
    # base_resolution: int = 64
    # finest_resolution: int = 512
    per_level_scale: int = 1.5
    interpolation: str = "Linear" #can be "Nearest", "Linear" or "Smoothstep", not used if not 'tcnn' encoder_type
    dim_in: int = len(image_shape)
    dim_hidden: int = 128 
    dim_out: int = 1
    num_layers: int = 4
    lr: float = 1e-3  # G requires training with a custom lr, usually lr * 0.1 
    interp_factor: int = 2

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")


config = BaseConfig()

mri_image = nib.load(config.image_path)
data = mri_image.get_fdata(dtype=np.float32)
data = data / data.max()
data = data[:,:,3,:]
config.image_shape = data.shape
config.dim_in = len(data.shape)

axes = []
for s in config.image_shape:
    axes.append(torch.linspace(0, 1, s))

mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

# points = mgrid[:,:,::2,:].reshape(-1, 3)
# values = data[..., ::2].reshape(-1, 1)
# xi = mgrid[:,:,1::2,:].reshape(-1, 3)

values = data[..., ::2]
itk_np = itk.GetImageFromArray(np.ascontiguousarray(values))


# interpolation = interpn(points, values, xi, method='linear')
# interpolator = LinearNDInterpolator(points, values)

itk_image = itk.imread(config.image_path)

#load image as itk object
    #create a itk image with correct spacing and stuff, will be fun I tell you

#create linear interpolator from itk
lin_interpolator = itk.LinearInterpolateImageFunction.New(itk_np)
# lin_interpolator.EvaluateAtContinuousIndex(index)

#interpolate
interpolated = np.zeros((352, 352, 15))
it = np.nditer(interpolated, flags=['multi_index'], op_flags=['readwrite'])
for i in it:
    interpolated[it.multi_index] = lin_interpolator.EvaluateAtContinuousIndex((it.multi_index[2] / 2, it.multi_index[1], it.multi_index[0]))

nib.save(nib.Nifti1Image(interpolated, affine=np.eye(4)), 'lightning_logs/version_55/itk_interpolated.nii.gz')