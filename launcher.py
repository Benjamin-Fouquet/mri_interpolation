"""
Launcher for trainings using datamodules and models
"""

import argparse
import copy
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing as proc

# import pytorch_lightning as pl
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from hydra.utils import call, get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from skimage import metrics
from torch import nn
from torch.nn import functional as F

# import functorch
from torchsummary import summary

from config import base
from datamodules import MriDataModule
from models import (
    HashMLP,
    HashSirenNet,
    ModulatedSirenNet,
    MultiSiren,
    PsfSirenNet,
    SirenNet,
)

torch.manual_seed(1337)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch size", type=int, required=False)
    parser.add_argument("--epochs", help="Number of epochs", type=int, required=False)
    parser.add_argument(
        "--accumulate_grad_batches",
        help="number of batches accumulated per gradient descent step",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--n_sample",
        help="number of points for psf in x, y, z",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--model_class", help="Modele class selection", type=str, required=False
    )
    parser.add_argument(
        "--enco_config_path",
        help="path for tinycuda encoding config",
        type=str,
        required=False,
    )
    args = parser.parse_args()


def export_to_txt(dict: dict, file_path: str = "") -> None:
    """
    Helper function to export dictionary to text file
    """
    with open(file_path + "config.txt", "a+") as f:
        for key in dict:
            f.write(str(key) + " : " + str(dict[key]) + "\n")


with open("config/hash_config.json") as f:
    enco_config = json.load(f)

if args.enco_config_path:
    with open(args.enco_config_path) as f:
        enco_config = json.load(f)

config = base.BaseConfig()

# parsed argument -> config
for key in args.__dict__:
    if args.__dict__[key] is not None:
        config.__dict__[key] = args.__dict__[key]

# correct for model class #could use getattr() here
if args.model_class is not None:
    if args.model_class == "PsfSirenNet":
        config.model_cls = PsfSirenNet
    elif args.model_class == "SirenNet":
        config.model_cls = SirenNet
    else:
        print("model class not recognized")
        raise ValueError

training_start = time.time()
####################
# MODEL DECLARATION#
####################
if config.checkpoint_path:
    model = config.model_cls.load_from_checkpoint(
        config.checkpoint_path,
        dim_in=config.dim_in,
        dim_hidden=config.dim_hidden,
        dim_out=config.dim_out,
        num_layers=config.num_layers,
        w0=config.w0,
        w0_initial=config.w0_initial,
        use_bias=config.use_bias,
        final_activation=config.final_activation,
        lr=config.lr,
        config=enco_config,
        n_frames=config.n_frames,
    )

else:

    model = config.model_cls(
        dim_in=config.dim_in,
        dim_hidden=config.dim_hidden,
        dim_out=config.dim_out,
        num_layers=config.num_layers,
        w0=config.w0,
        w0_initial=config.w0_initial,
        use_bias=config.use_bias,
        final_activation=config.final_activation,
        lr=config.lr,
        config=enco_config,
        n_frames=config.n_frames,
    )

#########################
# DATAMODULE DECLARATION#
#########################

datamodule = config.datamodule(config=config)
datamodule.prepare_data()
datamodule.setup()

train_loader = datamodule.train_dataloader()
# mean_train_loader = datamodule.mean_dataloader()
test_loader = datamodule.test_dataloader()

###############
# TRAINING LOOP#
###############

trainer = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=dict(config.accumulate_grad_batches)
    if config.accumulate_grad_batches
    else None,
    precision=16,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)
# trainer = pl.Trainer(gpus=config.device, max_epochs=config.epochs)
trainer.fit(model, train_loader)

training_stop = time.time()

filepath = model.logger.log_dir + "/"

image = nib.load(config.image_path)
data = image.get_fdata()
if config.dim_in == 2:
    data = data[:, :, int(data.shape[2] / 2)]
pred = torch.concat(trainer.predict(model, test_loader))

if config.dim_in == 3:
    output = pred.cpu().detach().numpy().reshape(data.shape)
    if output.dtype == "float16":
        output = np.array(output, dtype=np.float32)
    nib.save(
        nib.Nifti1Image(output, affine=image.affine), filepath + "2_16_0_result.nii.gz"
    )

if config.dim_in == 3:
    output = pred.cpu().detach().numpy().reshape(data.shape)
    if output.dtype == "float16":
        output = np.array(output, dtype=np.float32)
    nib.save(
        nib.Nifti1Image(output, affine=image.affine),
        filepath + "training_result.nii.gz",
    )
if config.dim_in == 4:
    output = np.zeros(config.image_shape)
    pred = pred.numpy()
    pred = np.array(pred, dtype=np.float32)
    for i in range(config.n_frames):
        im = pred[
            np.prod(config.image_shape[0:3])
            * i : np.prod(config.image_shape[0:3])
            * (i + 1),
            :,
        ]
        im = im.reshape(config.image_shape[0:3])
        output[..., i] = im
    nib.save(
        nib.Nifti1Image(output, affine=image.affine),
        filepath + "training_result.nii.gz",
    )

if config.dim_in == 2:
    output = pred.cpu().detach().numpy().reshape((data.shape[0], data.shape[1]))
    fig, axes = plt.subplots(1, 2)
    diff = data - output
    axes[0].imshow(output)
    axes[1].imshow(data)
    fig.suptitle("Standard training")
    axes[0].set_title("Prediction")
    axes[1].set_title("Ground truth")
    plt.savefig(filepath + "training_result_standard.png")
    plt.clf()

    plt.imshow(diff)
    plt.savefig(filepath + "difference.png")

config.export_to_txt(file_path=filepath)
export_to_txt(enco_config, file_path=filepath)

ground_truth = (data / np.max(data)) * 2 - 1

# try:
#     lats = model.get_latents()
#     for idx, lat in enumerate(lats):
#         torch.save(lat, filepath + f'lat{idx}.pt')
#     print('latents extracted')
# except:
#     print('No latents available')

# lats = lats[0]

# os.mkdir(filepath + 'latents/')

# create simple visualisation for latents, only usable with HashMLP
# for i in range(lats.shape[-1]):
#     lat = lats[:,i].detach().cpu().numpy()
#     lat = lat.reshape((config.image_shape))
#     lat = np.array(lat, dtype=np.float32)
#     # plt.imshow(lat[:,:,config.image_shape[-1] // 2])
#     plt.imshow(lat[0,:,:])
#     plt.savefig(filepath + f'latents/latent{i}.png')


# space upscaling
up_shape = (1408, 1408, 6)
loader = datamodule.upsampling(batch_size=100000, shape=up_shape)
upsample = torch.concat(trainer.predict(model, loader))
upsample = upsample.cpu().detach().numpy().reshape(up_shape)
if upsample.dtype == "float16":
    upsample = np.array(upsample, dtype=np.float32)
nib.save(
    nib.Nifti1Image(upsample, affine=np.eye(4)), filepath + "upsample_space.nii.gz"
)


# compare with normal interpolation
gt = nib.load(config.image_path)
up_affine = gt.affine[:, :].copy()
up_affine[0, 0:2] /= 4
up_affine[1, 0:2] /= 4
up_affine[2, 0:2] /= 4

up = proc.resample_from_to(gt, (up_shape, up_affine))
nib.save(
    up,
    "/home/aorus-users/Benjamin/git_repos/mri_interpolation/lightning_logs/version_23/"
    + "upsample_spline.nii.gz",
)

# #temporal upscaling
# up_shape = (352, 352, 6, 60)
# loader = datamodule.upsampling(batch_size=100000, shape=up_shape)
# upsample = torch.concat(trainer.predict(model, loader))
# upsample = upsample.cpu().detach().numpy().reshape(up_shape)
# if upsample.dtype == 'float16':
#     upsample = np.array(upsample, dtype=np.float32)
# nib.save(nib.Nifti1Image(upsample, affine=np.eye(4)), filepath + "upsample_time.nii.gz")

# # metrics
# with open(filepath + "scores.txt", "w") as f:
#     f.write("MSE : " + str(metrics.mean_squared_error(ground_truth, output)) + "\n")
#     f.write("PSNR : " + str(metrics.peak_signal_noise_ratio(ground_truth, output)) + "\n")
#     # if config.dim_in < 4:
#     #     f.write("SSMI : " + str(metrics.structural_similarity(ground_truth, output)) + "\n")
#     f.write(
#         "training time  : " + str(training_stop - training_start) + " seconds" + "\n"
#     )
#     f.write(
#         "Number of trainable parameters : "
#         + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
#         + "\n"
#     )  # remove condition if you want total parameters
#     f.write("Max memory allocated : " + str(torch.cuda.max_memory_allocated()) + "\n")

# difference between gt and pred as an image
nib.save(
    nib.Nifti1Image((ground_truth - output), affine=image.affine),
    filepath + "difference.nii.gz",
)


def export_to_txt(dict, file_path: str = "") -> None:
    with open(file_path + "config.txt", "a+") as f:
        for key in dict:
            f.write(str(key) + " : " + str(dict[key]) + "\n")


export_to_txt(enco_config, filepath)

# def latents_to_fig(lats):
#     fig, axes = plt.subplots(4, 4)
#     for i in range(4):
#         for j in range(4):
#             image = lats[0][:, i+j].detach().cpu().numpy().reshape(74, 74, 52)
#             axes[i, j].imshow(image[:,:,26], cmap='gray')
#     plt.savefig(filepath + 'latents_vis.png')

# latents_to_fig(lats)

# x = torch.linspace(0, 1, 290)
# y = torch.linspace(0, 1, 290)
# z = torch.linspace(0, 1, 10)

# mgrid = torch.stack(torch.meshgrid((x, y, z), indexing='ij'), dim=-1)

# X = mgrid.reshape(-1, 3)

# for i in range(10):
#     with torch.no_grad():
#         z = model.encoders[i](X.to('cuda'))
#         y_pred = model.decoder(z)
#         image = np.array(y_pred.detach().cpu().numpy(), dtype=np.float32).reshape(290, 290, 10)
#         nib.save(nib.Nifti1Image(image, affine=np.eye(4)), filepath + f'slice{i}.nii.gz')
