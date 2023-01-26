'''
Test code for a hashencoding and a siren inference for latent space
RESULT: gradient vanishing on siren, because the latent are very small
'''

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
from types import MappingProxyType

import matplotlib.pyplot as plt
# import torchio as tio
import nibabel as nib
# import pytorch_lightning as pl
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import commentjson as json  # TODO: inelegant, replace by hydra at a point
import tinycudann as tcnn
from datamodules import MriDataModule
from einops import rearrange
from models import Modulator, SirenNet
from utils import create_rn_mask

@dataclass
class BaseConfig:
    checkpoint_path = ''
    batch_size: int = 11151360 // 40  # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 1
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None #MappingProxyType({200: 2}) #MappingProxyType({0: 5})
    image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_shape = nib.load(image_path).shape
    hashconfig_path: str = 'config/hash_config.json'

    # Network parameters
    dim_in: int = 4
    dim_hidden: int = 512
    dim_out: int = 1
    num_layers: int = 2
    n_sample: int = 3
    w0: float = 30.0
    w0_initial: float = 30.0
    use_bias: bool = True
    final_activation = None
    lr: float = 1e-4  # G requires training with a custom lr, usually lr * 0.1
    datamodule: pl.LightningDataModule = MriDataModule

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")

config = BaseConfig()

with open("config/hash_config.json") as f:
    enco_config = json.load(f)

########################
# DATAMODULE DECLARATION#
########################
datamodule = config.datamodule(config=config)
datamodule.prepare_data()
datamodule.setup()

train_loader = datamodule.train_dataloader()
# mean_train_loader = datamodule.mean_dataloader()
test_loader = datamodule.test_dataloader()

encoder = tcnn.Encoding(
    n_input_dims=config.dim_in, encoding_config=enco_config["encoding"],
)
decoder = tcnn.Network(n_input_dims=enco_config["encoding"]["n_levels"] * enco_config["encoding"]["n_features_per_level"], n_output_dims=config.dim_out, network_config=enco_config['network'],)

model = torch.nn.Sequential(encoder, decoder)

siren = SirenNet(dim_in=enco_config["encoding"]['n_levels'] * enco_config["encoding"]['n_features_per_level'], dim_hidden=config.dim_hidden, dim_out=enco_config["encoding"]['n_levels'] * enco_config["encoding"]['n_features_per_level'], num_layers=config.num_layers)

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=1e-5)

model.to()

# manual training loop
losses = []

for epoch in range(config.epochs):

    # TRAINING LOOP
    for train_batch in train_loader:
        x, y = train_batch
        x = x.to("cuda")
        y = y.to("cuda")

        y_pred = model(x).float()

        loss = F.mse_loss(y_pred, y)
        print(f"epoch: {epoch}")
        print("train loss: ", loss.item())
        losses.append(loss.detach().cpu().numpy())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

pred = torch.zeros(1, 1)
for batch in test_loader:
    x, y = batch

    x = x.to("cuda")

    pred = torch.cat((pred, model(x).detach().cpu()))

pred = pred[1:, :].numpy()
pred= np.array(pred, dtype=np.float32)
pred = pred.reshape(config.image_shape)
nib.save(nib.Nifti1Image(pred, np.eye(4)), "out_hashMLP.nii.gz")

plt.plot(range(len(losses)), losses)
plt.savefig("losses_hashMLP.png")

optimizer_siren = torch.optim.Adam(params=siren.parameters(), lr=config.lr, weight_decay=1e-5)

siren.to('cuda')
siren = siren.half()

losses = []

for epoch in range(config.epochs):
    # TRAINING LOOP
    for train_batch in train_loader:
        x, y = train_batch
        x = x.to("cuda")
        y = y.to("cuda")
        y = y.half()

        lat = encoder(x)
        lat_pred = siren(lat)

        y_pred = decoder(lat_pred)

        loss = F.mse_loss(y_pred, y)
        print(f"epoch: {epoch}")
        print("train loss: ", loss.item())
        losses.append(loss.detach().cpu().numpy())

        loss.backward()
        optimizer_siren.step()
        optimizer_siren.zero_grad()

