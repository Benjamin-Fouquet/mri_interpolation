"""
Launcher for trainings using datamodules and models

TODO:
-normalisation is repeated and clunky, solve.
"""

import json
import os

import pytorch_lightning as pl
import torch

import hydra
from datamodules import MriDataModule
from hydra.utils import call, get_class, instantiate
from omegaconf import DictConfig, OmegaConf
import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union, FrozenSet, List

import matplotlib.pyplot as plt
import nibabel as nib
# import pytorch_lightning as pl
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from datamodules import MriDataModule
from einops import rearrange
from models import HashSirenNet, ModulatedSirenNet, PsfSirenNet, SirenNet, HashMLP, MultiSiren
# import functorch
from torchsummary import summary
from skimage import metrics
import time
import sys
from config import base
import glob

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

    args = parser.parse_args()

with open("config/hash_config.json") as f:
    enco_config = json.load(f)

config = base.BaseConfig()

# parsed argument -> config
for key in args.__dict__:
    if args.__dict__[key] is not None:
        config.__dict__[key] = args.__dict__[key]

# correct for model class
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
    model= config.model_cls().load_from_checkpoint(
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
        config=enco_config
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
        n_frames=config.n_frames
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
    accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
    precision=16,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)
# trainer = pl.Trainer(gpus=config.device, max_epochs=config.epochs)
trainer.fit(model, train_loader)

training_stop = time.time()

filepath = model.logger.log_dir + '/'

image = nib.load(config.image_path)
data = image.get_fdata()
if config.dim_in == 2:
    data = data[:, :, int(data.shape[2] / 2)]
pred = torch.concat(trainer.predict(model, test_loader))

if config.dim_in == 3:
    output = pred.cpu().detach().numpy().reshape(data.shape)
    if output.dtype == 'float16':
        output = np.array(output, dtype=np.float32)
    nib.save(
        nib.Nifti1Image(output, affine=np.eye(4)), filepath + "training_result.nii.gz"
    )
if config.dim_in == 4:
    output = np.zeros(config.image_shape)
    pred = pred.numpy()
    pred= np.array(pred, dtype=np.float32)
    for i in range(config.n_frames):
        im = pred[np.prod(config.image_shape[0:3]) * i: np.prod(config.image_shape[0:3]) * (i+1), :]
        im = im.reshape(config.image_shape[0:3])
        output[..., i] = im
    nib.save(
        nib.Nifti1Image(output, affine=np.eye(4)), filepath + "training_result.nii.gz"
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
    plt.savefig(filepath + "training_result_standart.png")
    plt.clf()

    plt.imshow(diff)
    plt.savefig(filepath + "difference.png")

config.export_to_txt(file_path=filepath)

ground_truth = (data / np.max(data))  * 2 - 1

# #space upscaling
# up_shape = (600, 600, 6, 15)
# loader = datamodule.upsampling(batch_size=100000, shape=up_shape)
# upsample = torch.concat(trainer.predict(model, loader))
# upsample = upsample.cpu().detach().numpy().reshape(up_shape)
# if upsample.dtype == 'float16':
#     upsample = np.array(upsample, dtype=np.float32)
# nib.save(nib.Nifti1Image(upsample, affine=np.eye(4)), filepath + "upsample_space.nii.gz")

# #temporal upscaling
# up_shape = (352, 352, 6, 60)
# loader = datamodule.upsampling(batch_size=100000, shape=up_shape)
# upsample = torch.concat(trainer.predict(model, loader))
# upsample = upsample.cpu().detach().numpy().reshape(up_shape)
# if upsample.dtype == 'float16':
#     upsample = np.array(upsample, dtype=np.float32)
# nib.save(nib.Nifti1Image(upsample, affine=np.eye(4)), filepath + "upsample_time.nii.gz")

# metrics
with open(filepath + "scores.txt", "w") as f:
    f.write("MSE : " + str(metrics.mean_squared_error(ground_truth, output)) + "\n")
    f.write("PSNR : " + str(metrics.peak_signal_noise_ratio(ground_truth, output)) + "\n")
    # if config.dim_in < 4:
    #     f.write("SSMI : " + str(metrics.structural_similarity(ground_truth, output)) + "\n")
    f.write(
        "training time  : " + str(training_stop - training_start) + " seconds" + "\n"
    )
    f.write(
        "Number of trainable parameters : "
        + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
        + "\n"
    )  # remove condition if you want total parameters
    f.write("Max memory allocated : " + str(torch.cuda.max_memory_allocated()) + "\n")

#difference between gt and pred as an image
nib.save(nib.Nifti1Image((ground_truth - output), affine=image.affine), filepath + 'difference.nii.gz')
