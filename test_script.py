"""
Barebone laucnher for tests

TODO:
-workers and device in conf OR include directly into class. prob if too many workers
-correct config
-normalisation is repeated and clunky, solve.
"""

import argparse
import glob
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, List, Tuple, Union

import matplotlib.pyplot as plt
import nibabel as nib
# import pytorch_lightning as pl
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

import hydra
import torchio as tio
from config import base
from datamodules import MNISTDataModule, MriDataModule, MriFramesDataModule
from einops import rearrange
from hydra.utils import call, get_class, instantiate
from models import (HashMLP, HashSirenNet, ModulatedSirenNet, MultiHashMLP,
                    MultiSiren, PsfSirenNet, SirenNet)
from omegaconf import DictConfig, OmegaConf
from skimage import metrics
from skimage.util import random_noise
# import functorch
from torchsummary import summary
import tinycudann as tcnn
import matplotlib.pyplot as plt

# # #build latent list
# lats = []
# lats_path = glob.glob('results/multi_MLP/lat*', recursive=True)
# for path in lats_path:
#     lats.append(torch.load(path))

# for idx, lat in enumerate(lats):
#     lat = lat.reshape(352, 352, 6, 15).cpu().detach().numpy()
#     lat = np.array(lat, dtype=np.float32)
#     nib.save(nib.Nifti1Image(lat, affine=np.eye(4)), f'results/latents_visualisation/lat_multiH_MultiMLP_{idx}.nii.gz')

torch.manual_seed(1337)

filepath = 'results/noRelu/'

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--batch_size", help="batch size", type=int, required=False)
#     parser.add_argument("--epochs", help="Number of epochs", type=int, required=False)
#     parser.add_argument(
#         "--accumulate_grad_batches",
#         help="number of batches accumulated per gradient descent step",
#         type=int,
#         required=False,
#     )
#     parser.add_argument(
#         "--n_sample",
#         help="number of points for psf in x, y, z",
#         type=int,
#         required=False,
#     )
#     parser.add_argument(
#         "--model_class", help="Modele class selection", type=str, required=False
#     )

#     args = parser.parse_args()

with open("config/hash_config.json") as f:
    enco_config = json.load(f)

@dataclass
class BaseConfig:
    checkpoint_path = None
    batch_size: int =  743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 500
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None #MappingProxyType({200: 2}) #MappingProxyType({0: 5})
    # image_path: str = 'data/t2_64cube.nii.gz'
    image_path: str = 'data/equinus_downsampled.nii.gz'
    image_shape = nib.load(image_path).shape
    hashconfig_path: str = 'config/hash_config.json'

    # Network parameters
    dim_in: int = 3
    dim_hidden: int = 128
    dim_out: int = 1
    num_layers: int = 4
    n_sample: int = 3
    w0: float = 30.0
    w0_initial: float = 30.0
    use_bias: bool = True
    final_activation = None
    lr: float = 1e-3  # G requires training with a custom lr, usually lr * 0.1
    datamodule: pl.LightningDataModule = MriDataModule
    model_cls: pl.LightningModule = HashMLP  
    # datamodule: pl.LightningDataModule = MriFramesDataModule
    # model_cls: pl.LightningModule = MultiHashMLP  
    n_frames: int = 15

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")

config = BaseConfig()

# # parsed argument -> config
# for key in args.__dict__:
#     if args.__dict__[key] is not None:
#         config.__dict__[key] = args.__dict__[key]

# # correct for model class
# if args.model_class is not None:
#     if args.model_class == "PsfSirenNet":
#         config.model_cls = PsfSirenNet
#     elif args.model_class == "SirenNet":
#         config.model_cls = SirenNet
#     else:
#         print("model class not recognized")
#         raise ValueError

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

class MLP(pl.LightningModule):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, lr, config, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features=dim_in, out_features=dim_hidden))
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_features=dim_hidden, out_features=dim_hidden))
        self.layers.append(nn.Linear(in_features=dim_hidden, out_features=dim_out))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=self.lr ,weight_decay=1e-5)   

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        loss = F.mse_loss(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        return y_pred

class HashMLP(pl.LightningModule):
    '''
    Lightning module for HashMLP. Ala mano decoder to see if results are comparable
    '''
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        config,
        lr,
        *args,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.lr = lr
        self.losses =[]
        self.lats = []

        self.encoder = tcnn.Encoding(n_input_dims=dim_in, encoding_config=config['encoding'])
        self.decoder= MLP(dim_in=self.config["encoding"]["n_levels"] * self.config["encoding"]["n_features_per_level"], dim_hidden=dim_hidden, dim_out=dim_out, num_layers=num_layers, lr=lr, config=config)
        # self.model = torch.nn.Sequential(self.encoder, self.decoder)


    def forward(self, x):
        z = self.encoder(x)
        y_pred = self.decoder(z)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr ,weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        y_pred = self.decoder(z)

        loss = F.mse_loss(y_pred, y)
        self.losses.append(loss.detach().cpu().numpy())

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        self.lats.append(z)
        y_pred = self.decoder(z)
        return y_pred

    def get_latents(self):
        return self.lats

# class Decoder(pl.LightningModule):
#     def __init__(self, dim_in, dim_hidden, dim_out, config, lr, *args: Any, **kwargs: Any) -> None:
#         super().__init__(*args, **kwargs)
#         self.lr = lr
#         self.model = tcnn.Network(n_input_dims=dim_in, n_output_dims=dim_out, network_config=config['network'])

#     def forward(self, x):
#         return self.model(x)

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.model.parameters(), lr=self.lr ,weight_decay=1e-5)   

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_pred = self(x)

#         loss = F.mse_loss(y_pred, y)
#         self.log("train_loss", loss)
#         return loss

#     def predict_step(self, batch, batch_idx):
#         x, y = batch
#         y_pred = self(x)
#         return y_pred
         
model = MLP(
    dim_in=config.dim_in,
    dim_hidden=config.dim_hidden,
    dim_out=config.dim_out,
    num_layers=config.num_layers,
    lr=config.lr,
    config=enco_config,
)

# model = Decoder(
#     dim_in=config.dim_in,
#     dim_hidden=config.dim_hidden,
#     dim_out=config.dim_out,
#     config=enco_config,
#     lr=config.lr,
# )
########################
#DATAMODULE DECLARATION#
########################
datamodule = config.datamodule(config=config)
datamodule.prepare_data()
datamodule.setup()

train_loader = datamodule.train_dataloader()
test_loader = datamodule.test_dataloader()

# model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr ,weight_decay=1e-5)       

losses = []

lats = []
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

ground_truth = nib.load(config.image_path)

pred = torch.concat(trainer.predict(model, test_loader))
if config.dim_in == 3:
    output = pred.cpu().detach().numpy().reshape(ground_truth.shape)
    if output.dtype == 'float16':
        output = np.array(output, dtype=np.float32)
    nib.save(
        nib.Nifti1Image(output, affine=np.eye(4)), filepath + "training_result.nii.gz"
    )
nib.save(nib.Nifti1Image(pred, affine=ground_truth.affine), "out.nii.gz") #file_map = ground_truth.file_map

output = pred #sugar syntaxing

gt_image = nib.load(config.image_path)

# #ugly reshaping, placeholder
# image = np.zeros(config.image_shape)
# pred = pred[1:, ...].numpy()
# pred= np.array(pred, dtype=np.float32)
# for i in range(config.n_frames):
#     im = pred[np.prod(config.image_shape[0:3]) * i: np.prod(config.image_shape[0:3]) * (i+1), :]
#     im = im.reshape(config.image_shape[0:3])
#     image[..., i] = im

# nib.save(nib.Nifti1Image(image, affine=gt_image.affine, header=gt_image.header), filepath + f"out_multiMLP{idx}.nii.gz") #file_map = ground_truth.file_map

plt.plot(range(len(losses)), losses)
plt.savefig(filepath + 'losses.png')

config.export_to_txt(file_path=filepath)

data = nib.load(config.image_path).get_fdata(dtype=np.float32)
ground_truth = (data / np.max(data))  * 2 - 1

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

'''
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


# # lat = torch.load('results/multi_hash_new/lat5.pt')
# for idx in range(lat.shape[-1]):
#     raw = lat[:,idx].reshape(352, 352, 6)
#     raw = np.array(raw, dtype=np.float32)
#     nib.save(nib.Nifti1Image(raw, affine=np.eye(4)), f'lightning_logs/version_289/lat_visu_per_dim_{idx}.nii.gz')


#add noise to images
# im1 = nib.load('data/equinus_downsampled.nii.gz')
# im2 = nib.load('/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz')

# data1 = im1.get_fdata(dtype=np.float64)
# data2 = im2.get_fdata(dtype=np.float64)

# data1 = data1 / np.max(data1)
# data2 = data2 / np.max(data2)

# noisy1 = random_noise(data1, mode='gaussian', var=0.1)
# noisy2 = random_noise(data2, mode='gaussian', var=0.01)

# nib.save(nib.Nifti1Image(noisy1, affine=im1.affine), 'data/equinus_singleframe_verynoisy.nii.gz')
# nib.save(nib.Nifti1Image(noisy2, affine=im2.affine), 'data/equinus_multiframe_noisy.nii.gz')

lats = torch.load(path)

tensor = lats.detach().cpu().numpy()
tensor = np.array(tensor, dtype=np.float32)

fig, axes = plt.subplots(4, 4, sharex='all', sharey='all')

for i in range(4):
    for j in range(4):
        image = tensor[:, (i + j)].reshape(352, 352, 6)
        image = image[:,:, 3] #select middle slice
        axes[i][j].imshow(image.T, cmap="gray") #cmap="gray", origin="lower"
        axes[i][j].axis('off')

plt.savefig('out.png')
plt.clf()

for i, slice in enumerate(zip(axes, tensor):
     axes[i].imshow(slice.T, origin="lower") #cmap="gray"
plt.show()
