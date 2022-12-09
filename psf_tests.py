'''
Small script to test siren vs map_coordinates
'''

import nibabel
import nibabel.processing
import numpy as np
from scipy.ndimage import map_coordinates
import models
import config as cg
import datamodules
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import os
import nibabel as nib
import functools
import asyncio
import numba
import SimpleITK as sitk
import scipy.ndimage

filepath = 'for_francois/correc/'

# model_file = 'lightning_logs/version_38/checkpoints/epoch=49-step=1800.ckpt'
model_file = '/home/aorus-users/Benjamin/git_repos/mri_interpolation/data/model_siren_e50_b20000.ckpt'

model = models.SirenNet().load_from_checkpoint(model_file, dim_in=3, dim_hidden=512, dim_out=1, num_layers=5, w0=30.)

config = cg.Config()

# HRimage = nibabel.load(config.image_path)
# LRimage = nibabel.processing.resample_to_output(HRimage, voxel_sizes=(0.7, 0.7, 2.1))

HRimage = nibabel.load('data/t2_111.nii.gz')
LRimage = nibabel.load('data/t2_113.nii.gz')
# trainer = pl.Trainer(gpus=[], max_epochs=10)


x = torch.linspace(-1, 1, LRimage.shape[0])
y = torch.linspace(-1, 1, LRimage.shape[1])
z = torch.linspace(-1, 1, LRimage.shape[2])

mgrid_lr = torch.stack(torch.meshgrid(x, y, z), dim=-1)

# x_flat = mgrid_lr.reshape(-1, 3)

# dataset = TensorDataset(x_flat, x_flat)
# loader = DataLoader(dataset, batch_size=300000, num_workers=os.cpu_count())

# yhat = torch.concat(trainer.predict(model, loader))

# pred = yhat.detach().cpu().numpy().reshape(LRimage.shape)
# nib.save(nib.Nifti1Image(pred, np.eye(4)), 'pred_lowres_2.nii.gz')

# #model declaration
# model = models.SirenNet(
#     dim_in=config.dim_in,
#     dim_hidden=config.dim_hidden,
#     dim_out=config.dim_out,
#     num_layers=config.num_layers,
#     w0=config.w0,
#     w0_initial=config.w0_initial,
#     use_bias=config.use_bias,
#     final_activation=config.final_activation,
# )

# datamodule = datamodules.MriDataModule(config=config)
# datamodule.prepare_data()
# datamodule.setup()
# train_loader = datamodule.train_dataloader()

# #training Siren on the high res
# opt = torch.optim.Adam(model.parameters(), lr=config.lr)
# trainer = pl.Trainer(gpus=config.device, max_epochs=config.epochs)
# trainer.fit(model, train_loader)

# Get image resolution
HRSpacing = np.float32(np.array(HRimage.header["pixdim"][1:4]))
LRSpacing = np.float32(np.array(LRimage.header["pixdim"][1:4]))

# Pre-compute PSF values
# PSF is a box centered around an observed pixel of LR image
# The size of the box is set as the size of a LR pixel (expressed in voxel space)
n_samples = 5
psf_sx = np.linspace(-0.5, 0.5, n_samples)
psf_sy = np.linspace(-0.5, 0.5, n_samples)
psf_sz = np.linspace(-0.5, 0.5, n_samples)

# Define a set of points for PSF values using meshgrid
# https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
psf_x, psf_y, psf_z = np.meshgrid(psf_sx, psf_sy, psf_sz, indexing="ij")

# Define gaussian kernel as PSF model
sigma = (
    1.0 / 2.3548
)  # could be anisotropic to reflect MRI sequences (see Kainz et al.)

def gaussian(x, sigma):
    return np.exp(-x * x / (2 * sigma * sigma))

psf = gaussian(psf_x, sigma) * gaussian(psf_y, sigma) * gaussian(psf_z, sigma)
psf = psf / np.sum(psf)

# Get data
HRdata = HRimage.get_fdata()
LRdata = LRimage.get_fdata()
outputdata = np.zeros(LRdata.shape)

# Define transforms
# This is where we could add slice-by-slice transform
LR_to_world = LRimage.affine
world_to_HR = np.linalg.inv(HRimage.affine)
LR_to_HR = world_to_HR @ LR_to_world

# PSF coordinates in LR image
psf_coordinates_in_LR = np.ones((4, psf.size))
psf_coordinates_in_siren = np.ones((3, psf.size))

# # Loop over LR pixels (i,j,k)
# for i in range(LRdata.shape[0]):
#     for j in range(LRdata.shape[1]):
#         for k in range(LRdata.shape[2]):

#             # coordinates of PSF box around current pixel
#             psf_coordinates_in_LR[0, :] = psf_x.flatten() + i
#             psf_coordinates_in_LR[1, :] = psf_y.flatten() + j
#             psf_coordinates_in_LR[2, :] = psf_z.flatten() + k

#             # Transform PSF grid to HR space
#             psf_coordinates_in_HR = LR_to_HR @ psf_coordinates_in_LR

#             # Get interpolated values at psf points in HR
#             interp_values = map_coordinates(
#                 HRdata,
#                 psf_coordinates_in_HR[0:3, :],
#                 order=0,
#                 mode="constant",
#                 cval=np.nan,
#                 prefilter=False,
#             )

#             # Compute new weigthed value of LR pixel
#             outputdata[i, j, k] = np.sum(psf.flatten() * interp_values)
# nibabel.save(nibabel.Nifti1Image(outputdata, LRimage.affine), 'map_coordinates_output.nii.gz')

# pred = torch.concat(trainer.predict(model, train_loader))
# nibabel.save(nibabel.Nifti1Image(pred.detach().reshape(260, 311, 260).numpy(), HRimage.affine), 'siren_highres_aftertraining.nii.gz')

#script part for testing the map_coordinates vs siren

#map high res coords to low res matrix, manual correction needed for 100% fit
# x = torch.linspace(0, HRimage.shape[0] - 1, LRimage.shape[0])
# y = torch.linspace(0, HRimage.shape[1] - 1, LRimage.shape[1])
# z = torch.linspace(0, HRimage.shape[2] - 3, LRimage.shape[2])
# mgrid = torch.stack(torch.meshgrid(x, y, z), dim=-1).numpy()

# high_res = HRimage.get_fdata(dtype=np.float32)

# output = map_coordinates(high_res, [mgrid[:,:,:,0], mgrid[:,:,:,1], mgrid[:,:,:,2]], mode='constant')

# d

# diff = output - LRimage.get_fdata(dtype=np.float32)
# nibabel.save(nibabel.Nifti1Image(diff, LRimage.affine), filepath + 'diff_ben_interp_gt.nii.gz')

# outputdata = np.zeros(LRdata.shape)
# for i in range(LRdata.shape[0]):
#     for j in range(LRdata.shape[1]):
#         for k in range(LRdata.shape[2]):
#         # # coordinates of PSF box around current pixel
#         #     psf_coordinates_in_LR[0, :] = psf_x.flatten() + i
#         #     psf_coordinates_in_LR[1, :] = psf_y.flatten() + j
#         #     psf_coordinates_in_LR[2, :] = psf_z.flatten() + k

#             # # Transform PSF grid to HR space
#             # psf_coordinates_in_HR = LR_to_HR @ psf_coordinates_in_LR

#             #Transform to Siren coordiantes (-1 to 1)
#             psf_coordinates_in_siren[0, :] = (psf_x.flatten() + i * 2) / 144 -1
#             psf_coordinates_in_siren[1, :] = (psf_y.flatten() + j * 2) / 144 -1
#             psf_coordinates_in_siren[2, :] = (psf_z.flatten() + k * 2) / 33 -1

#             x = torch.FloatTensor(psf_coordinates_in_siren.T)

#             interp_values = model(x)
#             interp_values = interp_values.detach().numpy()

#             outputdata[i, j, k] = np.sum(psf.flatten() * interp_values.T)

# nibabel.save(nibabel.Nifti1Image(outputdata, LRimage.affine), 'siren_interpol_ben_output.nii.gz')


#create psf mgrid, -1 extended by psf_spacing * 2 to accomodate for edge voxels
x = torch.linspace(-1.0056, 1.0056, (LRimage.shape[0]) * 5)
z = torch.linspace(-1.0056, 1.0056, (LRimage.shape[2]) * 5)
y = torch.linspace(-1.0242, 1.0242, (LRimage.shape[1]) * 5)

mgrid_psf = torch.stack(torch.meshgrid(x, y, z), dim=-1)

#create a TensorDataset and loader for prediction
dataset = TensorDataset(mgrid_psf.reshape(-1, 3), torch.zeros((len(mgrid_psf.reshape(-1, 3)), 1)))
loader = DataLoader(dataset=dataset, batch_size=300000, num_workers=config.num_workers // 2)

trainer = pl.Trainer(gpus=config.device)

yhat = torch.concat(trainer.predict(model, loader))

pred = yhat.detach().cpu().numpy().reshape(mgrid_psf.shape[:-1])

#TODO: calculate new affine = mat / 5
new_affine = LRimage.affine
new_affine[:,0:3] /= 5

# nib.save(nib.Nifti1Image(pred, new_affine), filepath + 'pred_before_PSF_bencode.nii.gz')

#convolve with PSF // for gradient maintenance also possible to do it with Conv3D, need 2 unsqueeze (channel and batch), weights can be set manually
def stride_conv3d(arr, arr2, s):
    return scipy.ndimage.convolve(arr, arr2)[::s, ::s, ::s]

convolved = stride_conv3d(pred, psf, 5)

# nib.save(nib.Nifti1Image(convolved, LRimage.affine), filepath + 'pred_after_PSF_bencode.nii.gz')


