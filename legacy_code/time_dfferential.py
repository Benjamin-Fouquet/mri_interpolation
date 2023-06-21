#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import argparse
import tinycudann as tcnn
import os

from typing import List, Optional, Union
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
import tinycudann as tcnn 
import torch
import pytorch_lightning as pl 
import torch.nn.functional as F
import json
import nibabel as nib 
from dataclasses import dataclass
import os
from types import MappingProxyType
import numpy as np
import math
import rff
import argparse
from torch.utils.tensorboard import SummaryWriter
from math import pi
import torch.utils.data
import matplotlib.pyplot as plt

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
    # Network parameters
    encoder_type: str = 'rff' #   
    n_frequencies: int = 32 if encoder_type == 'tcnn' else 352 #for classic, n_out = 2 * n_freq. For tcnn, n_out = 2 * n_freq * dim_in
    n_frequencies_t: int = 4 if encoder_type == 'tcnn' else 15
    dim_in: int = len(image_shape)
    dim_hidden: int = 256 
    dim_out: int = 1
    num_layers: int = 8
    skip_connections: tuple = () #(5, 11,)
    lr: float = 1e-3  # G requires training with a custom lr, usually lr * 0.1 
    interp_factor: int = 2

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch size", type=int, required=False)
    parser.add_argument("--epochs", help="Number of epochs", type=int, required=False)
    parser.add_argument("--image_path", help="path of image", type=str, required=False)
    parser.add_argument("--encoder_type", help="tcnn or classic", type=str, required=False)
    parser.add_argument("--n_frequencies", help="number of encoding frequencies", type=int, required=False)
    parser.add_argument("--n_frequencies_t", help="number of encoding frequencies for time", type=int, required=False)
    args = parser.parse_args()

def export_to_txt(dict: dict, file_path: str = "") -> None:
    '''
    Helper function to export dictionary to text file
    '''
    with open(file_path + "config.txt", "a+") as f:
        for key in dict:
            f.write(str(key) + " : " + str(dict[key]) + "\n")
            

class TimeDifferential(pl.LightningModule):
    '''
    Lightning module for HashMLP. 
    '''
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        n_layers,
        encoder_type,
        n_frequencies,
        n_frequencies_t,
        lr,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers 
        self.lr = lr
        self.n_frequencies = n_frequencies
        self.n_frequencies_t = n_frequencies_t
        self.encoder_type = encoder_type

        # self.encoder = tcnn.Encoding(n_input_dims=dim_in, encoding_config=config['encoding'])
        if self.encoder_type == 'tcnn': #if tcnn is especially required, set it TODO: getattr more elegant
            #create the dictionary
            self.encoder = tcnn.Encoding(n_input_dims=(self.dim_in), encoding_config= {"otype": "HashGrid", "n_levels": 8, "n_features_per_level": 2, "log2_hashmap_size": 19, "base_resolution": 16, "per_level_scale": 1.4, "interpolation": "Linear"}, dtype=torch.float32)
            # self.encoder = tcnn.Encoding(n_input_dims=(self.dim_in - 1), encoding_config={'otype': 'Frequency', 'n_frequencies': self.n_frequencies}, dtype=torch.float32)
            # self.encoder_t = tcnn.Encoding(n_input_dims=1, encoding_config={'otype': 'Frequency', 'n_frequencies': self.n_frequencies_t}, dtype=torch.float32)
        elif self.encoder_type == 'rff': #fallback to classic
            # self.encoder = rff.layers.GaussianEncoding(sigma=10.0, input_size=(self.dim_in - 1), encoded_size=self.n_frequencies)
            self.encoder = rff.layers.GaussianEncoding(sigma=10.0, input_size=(self.dim_in), encoded_size=self.n_frequencies)
            # self.encoder_t = rff.layers.GaussianEncoding(sigma=10.0, input_size=1, encoded_size=self.n_frequencies_t)

            
        # self.encoding_dim_out = (self.n_frequencies * 2 * (self.dim_in - 1) + self.n_frequencies_t * 2) if isinstance(self.encoder, tcnn.Encoding) else (self.n_frequencies * 2 + self.n_frequencies_t * 2)
        self.encoding_dim_out = (self.n_frequencies * 2 * (self.dim_in)) if isinstance(self.encoder, tcnn.Encoding) else (self.n_frequencies * 2)
        # self.encoding_dim_out = 16
        # self.encoder = torch.nn.Sequential(Siren(dim_in=self.dim_in, dim_out=self.dim_in * 2 * config['encoding']['n_frequencies']), Siren(dim_in=self.dim_in * 2 * config['encoding']['n_frequencies'], dim_out=self.dim_in * 2 * config['encoding']['n_frequencies']))
        # self.decoder = tcnn.Network(n_input_dims=self.encoder.n_output_dims, n_output_dims=dim_out, network_config=config['network'])
        self.decoder = torch.nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                in_features = self.encoding_dim_out
            else:
                in_features = self.dim_hidden
            block = torch.nn.Sequential(
                torch.nn.Linear(in_features=in_features, out_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden),
                torch.nn.BatchNorm1d(num_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden), #you can do batchnorm 3D + 1D and cat after
                # torch.nn.ReLU()
                torch.nn.GELU()
            )
            self.decoder.append(block)
            

    def forward(self, x):
        # coords = x[:, :(self.dim_in - 1)]
        # t = x[:, -1].unsqueeze(-1)
        # if isinstance(self.encoder, GaussianFourierFeatureTransform):
        #     coords = coords.unsqueeze(-1).unsqueeze(-1)
        #     t = t.unsqueeze(-1).unsqueeze(-1)
        # x = torch.hstack((self.encoder(coords), self.encoder_t(t)))
        # if isinstance(self.encoder, GaussianFourierFeatureTransform):
        #     x = x.squeeze(-1).squeeze(-1)
        # skip = x.clone()
        x = self.encoder(x)
        for idx, layer in enumerate(self.decoder):
            x = layer(x)
        return x 

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5) #weight_decay=1e-5
        return self.optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)

        loss = F.mse_loss(y_pred, y)

        self.log("train_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return y_pred

config = BaseConfig()

# parsed argument -> config
for key in args.__dict__:
    if args.__dict__[key] is not None:
        config.__dict__[key] = args.__dict__[key]


# image_list = ['data/equinus_frames/frame0.nii.gz', 'data/equinus_frames/frame2.nii.gz', 'data/equinus_frames/frame4.nii.gz']
image_list = [f'data/equinus_frames/frame{i}.nii.gz' for i in range(15)]
n_images = len(image_list)

#enco_paramters
n_encodings = len(image_list)
n_t_outputs = 5
base_resolution = 32
n_levels = 8

# Read first image
print("Reading : " + image_list[0])
image = nib.load(image_list[0])
data = image.get_fdata()
data = data / data.max()

# Create grid
dim = 4
nx = data.shape[0]
ny = data.shape[1]
nz = data.shape[2]
nt = 1
nmax = np.max([nx, ny, nz])

x = torch.linspace(0, 1, steps=nx)
y = torch.linspace(0, 1, steps=ny)
z = torch.linspace(0, 1, steps=nz)
t = torch.linspace(0, 1, steps=nt)
print(t)

mgrid = torch.stack(torch.meshgrid(t, x, y, z, indexing="ij"), dim=-1)

# Convert to X=(x,y,z) and Y=intensity
X = torch.Tensor(mgrid.reshape(-1, dim))
Y = torch.Tensor(data.reshape(-1, 1))

# Normalize intensities 
Y = (Y - torch.min(Y)) / (torch.max(Y) - torch.min(Y)) #* 2 - 1

Yplus = torch.Tensor(nib.load(image_list[1]).get_fdata()).reshape(-1, 1)

# Normalize intensities 
Yplus = (Yplus - torch.min(Yplus)) / (torch.max(Yplus) - torch.min(Yplus)) #* 2 - 1

print(X.shape)
print(Y.shape)

# Pytorch dataloader
dataset = torch.utils.data.TensorDataset(X, Y, Yplus)
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=True,
)

model = TimeDifferential(dim_in=config.dim_in, dim_hidden=config.dim_hidden, dim_out=config.dim_out, n_layers=config.num_layers, lr=config.lr, n_frequencies=config.n_frequencies, n_frequencies_t=config.n_frequencies_t, encoder_type=config.encoder_type)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)

#training loop
for epoch in range(config.epochs):
    for batch in train_loader:
        x, y, yplus = batch
        diffy = model(x)
        loss = F.mse_loss((y + diffy), yplus)
        loss.backward()
        print(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        

try:
    filepath = model.logger.log_dir + '/'
except:
    count = int(len(os.listdir('lightning_logs')))
    os.mkdir(f'lightning_logs/version_{count}')
    filepath = f'lightning_logs/version_{(count)}/'

#create a prediction
pred = torch.zeros(1, 1)
with torch.no_grad():
    for batch in test_loader:
        pred = torch.concat((pred, model(batch[0])))
pred = pred[1:]     

            
im = pred.reshape((352, 352, 6))
im = im.detach().cpu().numpy()
im = np.array(im, dtype=np.float32)
if len(im.shape) == 2:
    plt.imshow(im.T)
    plt.savefig(filepath + 'pred.png')
else:
    nib.save(nib.Nifti1Image(im, affine=np.eye(4)), filepath + 'pred.nii.gz')
    nib.save(nib.Nifti1Image(im + data, affine=np.eye(4)), filepath + 'next_frame.nii.gz')

config.export_to_txt(file_path=filepath)
            
            



