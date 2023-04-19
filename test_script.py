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
import nibabel.processing as proc

torch.manual_seed(1337)

filepath = 'results/latents_vis/'
    
enco_config = {
	"encoding": {
		"otype": "HashGrid",
		"n_levels": 16,
		"n_features_per_level": 2, 
		"log2_hashmap_size": 19,
		"base_resolution": 16,
		"per_level_scale": 2,
		"interpolation": "Linear"
	},
	"network": {
		"otype": "FullyFusedMLP", 
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": 64,
		"n_hidden_layers": 2
	}
}

@dataclass
class BaseConfig:
    image_path: str = 'data/t2_64cube.nii.gz'
    # image_path: str = 'data/equinus_downsampled.nii.gz'
    image_shape = nib.load(image_path).shape
    checkpoint_path = None
    batch_size: int =  int(np.prod(image_shape)) # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 50
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None #MappingProxyType({200: 2}) #MappingProxyType({0: 5})
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
    n_frames: int = 1

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")

config = BaseConfig()

class MultiHashMLP(pl.LightningModule):
    '''
    Lightning module for MultiHashMLP. 
    Batch size = 1 means whole volume, setup this way as you need the frame idx
    '''
    def __init__(
        self,
        dim_in,
        dim_out,
        n_frames,
        config,
        lr,
        *args,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_frames = n_frames
        self.lr = lr
        self.losses =[]
        self.latents = []

        self.encoders = nn.ModuleList()
        for _ in range(self.n_frames):
            self.encoders.append(tcnn.Encoding(n_input_dims=dim_in, encoding_config=config['encoding']))
        self.decoder= tcnn.Network(n_input_dims=self.config["encoding"]["n_levels"] * self.config["encoding"]["n_features_per_level"], n_output_dims=dim_out, network_config=config['network'])

        # if torch.cuda.is_available():
        #     self.decoder.to('cuda')

        self.automatic_optimization = True #set to False if you need to propagate gradients manually. Usually lightning does a good job at no_grading models not used for a particular training step. Also, grads are not propagated in inctive leaves

    def forward(self, x, frame_idx):
        z =self.encoders[frame_idx](x)
        y_pred = self.decoder(z)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr ,weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y, frame_idx = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        z = self.encoders[frame_idx](x) #pred, model(x)
        y_pred = self.decoder(z)
        loss = F.mse_loss(y_pred, y)

        self.losses.append(loss.detach().cpu().numpy())

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        '''
        TODO: adapt for frame adaptive.
        '''
        x, y, frame_idx = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        z = self.encoders[frame_idx](x)
        self.latents.append(z)
        y_pred = self.decoder(z)
        return y_pred
    
    def get_latents(self):
        return self.latents

class HashMLP(pl.LightningModule):
    '''
    Lightning module for HashMLP.
    '''
    def __init__(
        self,
        dim_in,
        dim_out,
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
        self.decoder= tcnn.Network(n_input_dims=self.config["encoding"]["n_levels"] * self.config["encoding"]["n_features_per_level"], n_output_dims=dim_out, network_config=self.config)
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
        
model = HashMLP(dim_in=config.dim_in, dim_hidden=config.dim_hidden, dim_out=config.dim_out, num_layers=config.num_layers, lr=config.lr, config=enco_config)
########################
#DATAMODULE DECLARATION#
########################
datamodule = config.datamodule(config=config)
datamodule.prepare_data()
datamodule.setup()

train_loader = datamodule.train_dataloader()
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

training_start = time.time()

trainer.fit(model, train_loader)

pred = torch.concat(trainer.predict(model, test_loader))

training_stop = time.time()

ground_truth = nib.load(config.image_path)

im_pred = pred.detach().cpu().numpy().reshape(config.image_shape)
im_pred = np.array(im_pred, dtype=np.float32)

nib.save(nib.Nifti1Image(im_pred, affine=ground_truth.affine), filepath + 'prediction.nii.gz')


model2 = MultiHashMLP(dim_in=config.dim_in, dim_hidden=config.dim_hidden, dim_out=config.dim_out, num_layers=config.num_layers, lr=config.lr, n_frames=config.n_frames, config=enco_config)

config = {
	"encoding": {
		"otype": "HashGrid",
		"n_levels": 4,
		"n_features_per_level": 1, 
		"log2_hashmap_size": 19,
		"base_resolution": 32,
		"per_level_scale": 2,
		"interpolation": "Linear"
	},
	"network": {
		"otype": "CutlassMLP", 
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": 32,
		"n_hidden_layers": 1
	}
    
}

net1 = tcnn.Network(n_input_dims=20, n_output_dims=1, network_config=config['network'])
net2 = torch.nn.ModuleList((torch.nn.Linear(in_features=3, out_features=64, bias=True), torch.nn.ReLU(), torch.nn.Linear(in_features=64, out_features=1, bias=True), torch.nn.ReLU()))

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    

count_parameters(net1)
count_parameters(net2)

# def lat_reshape(latent):

# for i in range(lats.shape[-1]):
#     lat = lats[:,i].detach().cpu().numpy()
#     lat = lat.reshape((16, 16, 16))
#     lat = np.array(lat, dtype=np.float32)
#     plt.imshow(lat[:,:,8])
#     plt.savefig(f'results/latents_vis/{i}_lowres.png')
    
fig, axes = plt.subplots(8, 4)
for i in range(8):
    for j in range(4):
        lat = lats[:, (i + 1 * j + 1) - 1].detach().cpu().numpy()
        lat = lat.reshape((64, 64, 64))
        lat = np.array(lat, dtype=np.float32)    
        axes[i, j].imshow(lat[:,:,32])
        print('pouet')

plt.savefig('lightning_logs/version_379/latents/all_lats.png')

enco_config = {
	"encoding": {
		"otype": "HashGrid",
		"n_levels": 16,
		"n_features_per_level": 1, 
		"log2_hashmap_size": 19,
		"base_resolution": 8,
		"per_level_scale": 2,
		"interpolation": "Linear"
	},
	"network": {
		"otype": "FullyFusedMLP", 
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": 64,
		"n_hidden_layers": 1
	}
}

@dataclass
class BaseConfig:
    checkpoint_path = None
    image_path: str = 'data/t2_64cube.nii.gz'
    image_shape = nib.load(image_path).shape
    batch_size: int = int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 100
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None #MappingProxyType({200: 2}) #MappingProxyType({0: 5})
    # image_path: str = 'data/equinus_frames/frame8.nii.gz'
    # image_path: str = '/mnt/Data/DHCP/sub-CC00074XX09_ses-28000_desc-restore_T2w.nii.gz'
    # image_path:str = '/mnt/Data/HCP/HCP100_T1T2/146432_T2.nii.gz'
    
    # image_path: str = '/home/aorus-users/Benjamin/git_repos/mri_interpolation/data/equinus_sameframes.nii.gz'
    # image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    # image_path: str = 'data/equinus_singleframe_noisy.nii.gz'
    coordinates_spacing: np.array = np.array(
        (2 / image_shape[0], 2 / image_shape[1], 2 / image_shape[2])
    )
    hashconfig_path: str = 'config/hash_config.json'

    # Network parameters
    dim_in: int = len(image_shape)
    dim_hidden: int = 64
    dim_out: int = 1
    num_layers: int = 6
    n_sample: int = 3
    w0: float = 30.0
    w0_initial: float = 30.0
    use_bias: bool = True
    final_activation = None
    lr: float = 5e-3  # G requires training with a custom lr, usually lr * 0.1
    datamodule: pl.LightningDataModule = MriDataModule
    model_cls: pl.LightningModule = HashMLP  
    # datamodule: pl.LightningDataModule = MriFramesDataModule
    # model_cls: pl.LightningModule = MultiHashMLP  
    n_frames: int = image_shape[-1] if len(image_shape) == 4 else None

    # # output
    # output_path: str = "results_hash/"
    # if os.path.isdir(output_path) is False:
    #     os.mkdir(output_path)
    # experiment_number: int = 0 if len(os.listdir(output_path)) == 0 else len(
    #     os.listdir(output_path)
    # )

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")

config = BaseConfig()

model = HashMLP(dim_in=config.dim_in, dim_hidden=config.dim_hidden, dim_out=config.dim_out, num_layers=config.num_layers, lr=config.lr, config=enco_config)

model = HashMLP.load_from_checkpoint('lightning_logs/version_384/checkpoints/epoch=99-step=100.ckpt', dim_in=config.dim_in, dim_hidden=config.dim_hidden, dim_out=config.dim_out, num_layers=config.num_layers, lr=config.lr, config=enco_config)

model.eval()

for i in range(64):
    for j in range(64):
            for k in range(64):
                    kij[k, i, j] = image[i, j, k]
                    
     
     
     
     
     
import numpy as np 
import torch     

def hashing(coords):
    hashtable = {}
    pi = [1, 2654435761]
    T = 2 ** 19
    for x, y in coords:
        key = 
                         
X = np.array([i for i in range(64)])
Y = np.array([i for i in range(64)])

grid = np.stack((X, Y)).T

#list of primes for dimension agnostic algorithme
p1 = 1
p2 = 2654435761
T = 2 ** 19

#how to downsample ? Tried a pure translation from paper, for a given level
#TODO: where is lowest res in this algo ? retry with proper form, then prove that duplicates happend at lower res
uniques = []

Nmax = 16
Nmin = 1

for i in range(Nmax):

    L = i + 2
    b = (np.log(Nmax) - np.log(Nmin)) / (L - 1)
    Nl = Nmin * b
    grid_l = grid * Nl


    hashtable = {}

    for idx, (x, y) in enumerate(grid_l):
        hashtable[idx] = int(x) ^ 1 % T
        hashtable[idx] = int(y) ^ p2 % T
        

    uniques.append(len(np.unique(hashtable)))
    
    #parameters organized in L * F levels. 
    grid = torch.FloatTensor(200000, 16)
    params= torch.nn.parameter.Parameter(grid)
    torch.nn.ParameterList
    torch.nn.parameter.Parameter
    torch.nn.Linear
    
    class TinyEncoding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.grid = nn.parameter.Parameter(torch.FloatTensor(image_shape, n_levels * n_features))
            


for i in range(15):
    lat = torch.load(f'lightning_logs/version_0/lat{i}.pt')
    lat = lat.cpu().numpy()
    lat = np.array(lat, dtype=np.float32)
    
    im0 = lat[:,0].reshape(352, 352, 6)
    im6 = lat[:,6].reshape(352, 352, 6)
    im12 = lat[:,12].reshape(352, 352, 6)
    
    nib.save(nib.Nifti1Image(im0, affine=np.eye(4)), f'lightning_logs/version_0/latents/frame{i}_lat0.nii.gz')    
    nib.save(nib.Nifti1Image(im6, affine=np.eye(4)), f'lightning_logs/version_0/latents/frame{i}_lat6.nii.gz')
    nib.save(nib.Nifti1Image(im12, affine=np.eye(4)), f'lightning_logs/version_0/latents/frame{i}_lat12.nii.gz')
    
    

frames = glob.glob('lightning_logs/version_0/latents/frame*_lat0.nii.gz', recursive=True)

fig, axes = plt.subplots(2, 5)
for idx, frame in enumerate(frames):
    i = idx // 5
    j = idx % 5
    image = nib.load(frame).get_fdata()
    # axes[i][j].imshow(image[:,:,3].T, origin="lower") #cmap="gray"
    axes[i][j].imshow(image.T[3,:,:], origin="lower", cmap='gray')

plt.savefig('lightning_logs/version_0/latents/lat0.png')

frames = glob.glob('lightning_logs/version_0/latents/frame*_lat6.nii.gz', recursive=True)

fig, axes = plt.subplots(2, 5)
for idx, frame in enumerate(frames):
    i = idx // 5
    j = idx % 5
    image = nib.load(frame).get_fdata()
    # axes[i][j].imshow(image[:,:,3].T, origin="lower") #cmap="gray"
    axes[i][j].imshow(image.T[3,:,:], origin="lower", cmap='gray')

plt.savefig('lightning_logs/version_0/latents/lat6.png')

frames = glob.glob('lightning_logs/version_0/latents/frame*_lat12.nii.gz', recursive=True)

fig, axes = plt.subplots(2, 5)
for idx, frame in enumerate(frames):
    i = idx // 5
    j = idx % 5
    image = nib.load(frame).get_fdata()
    # axes[i][j].imshow(image[:,:,3].T, origin="lower") #cmap="gray"
    axes[i][j].imshow(image.T[3,:,:], origin="lower", cmap='gray')

plt.savefig('lightning_logs/version_0/latents/lat12.png')


image = nib.load('/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz')
stack = np.zeros((96, 96, 6, 1))
for i in range(image.shape[-1]):
    reshaped_image = proc.conform((image.slicer[..., i]), (96, 96, 6))
    reshaped_data = reshaped_image.get_fdata().reshape(96, 96, 6, 1)
    stack = np.concatenate((stack, reshaped_data), axis=-1)
    
nib.save(nib.Nifti1Image(stack[..., 1:], affine=reshaped_image.affine), 'data/tiny_equinus.nii.gz')

pred10 = nib.load('lightning_logs/version_10/pred_spe.nii.gz') #frame 0 and 6 with 10 interps
pred11 = nib.load('lightning_logs/version_11/pred_spe.nii.gz') #frame 0 and 1 with 10 interps
pred12 = nib.load('lightning_logs/version_12/pred_spe.nii.gz') #all frames, 45 interps
pred13 = nib.load('lightning_logs/version_13/pred_spe.nii.gz') #frame 0, 2, 4, 5 interps, to be compared with frame 1 and 3

#load ground truth data...when fucking possible
frames = [nib.load(f'data/equinus_frames/frame{i}.nii.gz') for i in range(15)]

fig, axes = plt.subplots(3, 5)
for i in range(5):
    fr = frames[i].get_fdata()
    fr = (fr - np.min(fr)) / (np.max(fr) - np.min(fr)) * 2 - 1
    axes[0][i].imshow(fr[..., 3].T, origin="lower", cmap="gray") #cmap="gray"
    sl = images[i]
    axes[1][i].imshow(sl[..., 3].T, origin="lower", cmap="gray") #cmap="gray"
    diff = fr - sl
    axes[2][i].imshow(diff[..., 3].T, origin="lower", cmap="gray") #cmap="gray"

plt.savefig('interp_frame_0_5_comparison3.png')  

#pred
for i in range(5):
    sl = pred13.slicer[..., i].get_fdata()
    axes[1][i].imshow(sl[..., 3].T, origin="lower", cmap="gray") #cmap="gray"
        
plt.savefig('interp_frame_0_5.png')  


fig, axes = plt.subplots(7, 5)    
for i in range(7):
    for j in range(5):
        sl = pred12.slicer[..., (j * (i + 1)) + j].get_fdata()
        axes[i][j].imshow(sl[..., 3].T, origin="lower", cmap="gray") #cmap="gray"
plt.savefig('interp_frame_all.png')


lat0 = torch.load('lightning_logs/version_19/lat0.pt')
lat2 = torch.load('lightning_logs/version_19/lat2.pt')
lat4 = torch.load('lightning_logs/version_19/lat4.pt')

lat1_interp = (lat0 + lat2) / 2
lat3_interp = (lat2 + lat4) / 2

config = BaseConfig()

#reinterpret latents using decoder, not possible because tinycuda shit
model = MultiHashMLP.load_from_checkpoint('lightning_logs/version_19/checkpoints/epoch=499-step=7500.ckpt',
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
        n_frames=config.n_frames)

output = pred.cpu().detach().numpy().reshape(image.shape)
if output.dtype == 'float16':
    output = np.array(output, dtype=np.float32)
nib.save(
    nib.Nifti1Image(output, affine=np.eye(4)), "interp_3.nii.gz"
    )

images = []
for lat in preds:
    y_pred = net.mlp(lat)
    output = y_pred.cpu().detach().numpy().reshape(image.shape)
    output = np.array(output, dtype=np.float32)
    images.append(output)
    
    
#mgrid_fonction

    
