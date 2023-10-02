'''
Short interpolation script for itk interpolation as baseline
'''''

import torch
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

image_path = ''

mri_image = nib.load(image_path)
data = mri_image.get_fdata(dtype=np.float32)
data = data / data.max()
data = data[:,:,3,:]

axes = []
for s in data.shape:
    axes.append(torch.linspace(0, 1, s))

mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

values = data[..., ::2]

itk_np = itk.GetImageFromArray(np.ascontiguousarray(values))

itk_image = itk.imread(image_path)

#create linear interpolator from itk
lin_interpolator = itk.LinearInterpolateImageFunction.New(itk_np)

#interpolate
interpolated = np.zeros(data.shape)
it = np.nditer(interpolated, flags=['multi_index'], op_flags=['readwrite'])
for i in it:
    interpolated[it.multi_index] = lin_interpolator.EvaluateAtContinuousIndex((it.multi_index[2] / 2, it.multi_index[1], it.multi_index[0]))

nib.save(nib.Nifti1Image(interpolated, affine=np.eye(4)), 'itk_interpolated.nii.gz')