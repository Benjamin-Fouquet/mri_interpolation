"""
Short interpolation script for itk interpolation as baseline
""" ""

import argparse
import os
from dataclasses import dataclass
from math import pi
from types import MappingProxyType

import itk
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.utils.data
from skimage import metrics
from torch.utils.tensorboard import SummaryWriter

import encoding

image_path = ""

mri_image = nib.load(image_path)
data = mri_image.get_fdata(dtype=np.float32)
data = data / data.max()
data = data[:, :, 3, :]

axes = []
for s in data.shape:
    axes.append(torch.linspace(0, 1, s))

mgrid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)

values = data[..., ::2]

itk_np = itk.GetImageFromArray(np.ascontiguousarray(values))

itk_image = itk.imread(image_path)

# create linear interpolator from itk
lin_interpolator = itk.LinearInterpolateImageFunction.New(itk_np)

# interpolate
interpolated = np.zeros(data.shape)
it = np.nditer(interpolated, flags=["multi_index"], op_flags=["readwrite"])
for i in it:
    interpolated[it.multi_index] = lin_interpolator.EvaluateAtContinuousIndex(
        (it.multi_index[2] / 2, it.multi_index[1], it.multi_index[0])
    )

nib.save(nib.Nifti1Image(interpolated, affine=np.eye(4)), "itk_interpolated.nii.gz")
