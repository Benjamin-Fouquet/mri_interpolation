"""
Barebone laucnher for tests

TODO:
-workers and device in conf OR include directly into class. prob if too many workers
-correct config
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
from models import HashSirenNet, ModulatedSirenNet, PsfSirenNet, SirenNet, HashMLP
# import functorch
from torchsummary import summary
from skimage import metrics
import time
import sys
from config import base


torch.manual_seed(1337)

filepath = 'results/multi_hash/'

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

########################
# DATAMODULE DECLARATION#
########################
datamodule = config.datamodule(config=config)
datamodule.prepare_data()
datamodule.setup()

train_loader = datamodule.train_dataloader()
# mean_train_loader = datamodule.mean_dataloader()
test_loader = datamodule.test_dataloader()

model.to('cuda')

enc_optimizers = []
for i in range(config.n_frames):
    optimizer = torch.optim.Adam(model.encoders[i].parameters(), lr=config.lr ,weight_decay=1e-5)
    enc_optimizers.append(optimizer)

dec_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=config.lr ,weight_decay=1e-5)       

losses = []

lats = []
###############
# TRAINING LOOP#
###############
for epoch in range(config.epochs):
    # TRAINING LOOP
    for train_batch in train_loader:
        x, y, frame_idx = train_batch
        x = x.to("cuda").squeeze(0)
        y = y.to("cuda").squeeze(0)

        lat = model.encoders[frame_idx](x)
        if epoch == config.epochs - 1:
            lats.append(lat.detach().cpu().numpy())
        z = model.decoder(lat)
        z = z.float()

        loss = F.mse_loss(z, y)
        print(f"epoch: {epoch}")
        print("train loss: ", loss.item())
        losses.append(loss.detach().cpu().numpy())

        enc_optimizers[frame_idx].zero_grad()
        dec_optimizer.zero_grad()

        loss.backward()
        enc_optimizers[frame_idx].step()
        dec_optimizer.step()


training_stop = time.time()

pred = torch.zeros(1, 1)
for batch in test_loader:
    x, y, frame_idx = batch
    x = x.to("cuda").squeeze(0)
    pred = torch.cat((pred, model(x, frame_idx).detach().cpu()))

# pred = pred[1:, ...].numpy()
# pred = pred.reshape(config.image_shape)
# pred= np.array(pred, dtype=np.float32)
# nib.save(nib.Nifti1Image(pred, affine=ground_truth.affine, header=ground_truth.header, extra=ground_truth.extra, file_map=ground_truth.file_map), "out_multihash.nii.gz") #file_map = ground_truth.file_map

gt_image = nib.load(config.image_path)

#ugly reshaping, placeholder
image = np.zeros(config.image_shape)
pred = pred[1:, ...].numpy()
pred= np.array(pred, dtype=np.float32)
for i in range(config.n_frames):
    im = pred[np.prod(config.image_shape[0:3]) * i: np.prod(config.image_shape[0:3]) * (i+1), :]
    im = im.reshape(config.image_shape[0:3])
    image[..., i] = im

nib.save(nib.Nifti1Image(image, affine=gt_image.affine, header=gt_image.header), filepath + "out_multihash.nii.gz") #file_map = ground_truth.file_map

plt.plot(range(len(losses)), losses)
plt.savefig(filepath +'losses_MultiHash.png')

config.export_to_txt(file_path=filepath)

data = nib.load(config.image_path).get_fdata(dtype=np.float64)
ground_truth = (data / np.max(data))  * 2 - 1

output = image

# ###############
# #INTERPOLATION#
# ###############
# #create one latent per frame
# lats = []
# for batch in test_loader:
#     x, y, frame_idx = batch
#     x = x.to("cuda").squeeze(0)
#     lat = model.encoders[frame_idx](x)
#     lats.append(lat.detach())

# #interp latents at half distance
# interps = []
# for i in range(len(lats) - 1):
#     interps.append(0.5 * lats[i] + 0.5 * lats[i + 1])

# #recontruct using the MLP
# pred = torch.zeros(1, 1)
# for interp in interps:
#     pred = torch.cat((pred, model.decoder(interp).detach().cpu()))

# #save results
# image = np.zeros(config.image_shape)
# pred = pred[1:, ...].numpy()
# pred= np.array(pred, dtype=np.float32)
# for i in range(len(interps)):
#     im = pred[np.prod(config.image_shape[0:3]) * i: np.prod(config.image_shape[0:3]) * (i+1), :]
#     im = im.reshape(config.image_shape[0:3])
#     image[..., i] = im

# nib.save(nib.Nifti1Image(image, affine=gt_image.affine, header=gt_image.header), filepath + "out_interp.nii.gz") 

#Save lats
for idx, lat in enumerate(lats):
    torch.save(lat, filepath + f'lat{idx}.pt')

#save decoder
state_decoder = model.decoder.state_dict()

torch.save(model.decoder.state_dict(), filepath+'decoder_statedict.pt')

# metrics
with open(filepath + "scores.txt", "w") as f:
    f.write("MSE : " + str(metrics.mean_squared_error(ground_truth, output)) + "\n")
    f.write("PSNR : " + str(metrics.peak_signal_noise_ratio(ground_truth, output)) + "\n")
    if config.dim_in < 4:
        f.write("SSMI : " + str(metrics.structural_similarity(ground_truth, output)) + "\n")
    f.write(
        "training time  : " + str(training_stop - training_start) + " seconds" + "\n"
    )
    f.write(
        "Number of trainable parameters : "
        + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
        + "\n"
    )  # remove condition if you want total parameters
    f.write("Max memory allocated : " + str(torch.cuda.max_memory_allocated()) + "\n")

nib.save(nib.Nifti1Image((ground_truth - output), affine=gt_image.affine), filepath + 'difference.nii.gz')

# #space upscaling
# up_shape = (600, 600, 6, 15)
# loader = datamodule.upsampling(shape=up_shape)
# upsample = torch.zeros(1, 1)
# for batch in loader:
#     x, y, frame_idx = batch
#     x = x.to("cuda").squeeze(0)
#     upsample = torch.cat((upsample, model(x, frame_idx).detach().cpu()))


# image = np.zeros(up_shape)
# upsample = upsample[1:, ...].numpy()
# upsample= np.array(upsample, dtype=np.float32)
# for i in range(config.n_frames):
#     im = upsample[np.prod(up_shape[0:3]) * i: np.prod(up_shape[0:3]) * (i+1), :]
#     im = im.reshape(up_shape[0:3])
#     image[..., i] = im

# nib.save(nib.Nifti1Image(image, affine=ground_truth.affine, header=ground_truth.header), "out_upspatial_multihash.nii.gz") #file_map = ground_truth.file_map

# #temporal upscaling, not possible due to indexing of encoders
# up_shape = (352, 352, 6, 60)
# loader = datamodule.upsampling(shape=up_shape)
# upsample = torch.zeros(1, 1)
# for batch in loader:
#     x, y, frame_idx = batch
#     x = x.to("cuda").squeeze(0)
#     upsample = torch.cat((upsample, model(x, frame_idx).detach().cpu()))


# image = np.zeros(up_shape)
# upsample = upsample[1:, ...].numpy()
# upsample= np.array(upsample, dtype=np.float32)
# for i in range(config.n_frames):
#     im = upsample[np.prod(up_shape[0:3]) * i: np.prod(up_shape[0:3]) * (i+1), :]
#     im = im.reshape(up_shape[0:3])
#     image[..., i] = im

# nib.save(nib.Nifti1Image(image, affine=ground_truth.affine, header=ground_truth.header), "out_uptemporal_multihash.nii.gz") #file_map = ground_truth.file_map

'''
print('pouet')

#load both images
gt_image = nib.load('/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz')
pred = nib.load('lightning_logs/version_232/training_result.nii.gz')

ground_truth = gt_image.get_fdata()
ground_truth = ground_truth / np.max(ground_truth) * 2 - 1
output = pred.get_fdata()

with open("scores_per_frame_sameNet.txt", "w") as f:
    for i in range(gt_image.shape[-1]):
        f.write(f"PSNR_{i} : " + str(metrics.peak_signal_noise_ratio(ground_truth[:,:,:,i], output[:,:,:,i])) + "\n")

'''


