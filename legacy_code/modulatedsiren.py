"""
Pure pytorch version of modulated siren, made for early tests
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

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

import commentjson as json
import config as cf  # TODO: inelegant, replace by hydra at a point
import tinycudann as tcnn
from datamodules import MriDataModule
from einops import rearrange
from models import Modulator, SirenNet
from utils import create_rn_mask


@dataclass
class Config:
    checkpoint_path = ""
    batch_size: int = 16777216 // 50  # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 100
    num_workers: int = os.cpu_count()
    # num_workers:int = 0
    device = [0] if torch.cuda.is_available() else []
    # device = []
    # accumulate_grad_batches = {200: 5, 250: 10, 300: 20, 350: 40}
    accumulate_grad_batches = None
    image_path: str = "data/t2_256cube.nii.gz"
    image_shape = nib.load(image_path).shape
    coordinates_spacing: np.array = np.array(
        (2 / image_shape[0], 2 / image_shape[1], 2 / image_shape[2])
    )

    # Network parameters
    dim_in: int = 3
    dim_hidden: int = 256
    dim_out: int = 1
    num_layers: int = 5
    n_sample: int = 3
    w0: float = 30.0
    w0_initial: float = 30.0
    use_bias: bool = True
    final_activation = None
    lr: float = 1e-4  # G requires training with a custom lr, usually lr * 0.1
    datamodule: pl.LightningDataModule = MriDataModule

    comment: str = ""

    # output
    output_path: str = "results_siren/"
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path)
    experiment_number: int = 0 if len(os.listdir(output_path)) == 0 else len(
        os.listdir(output_path)
    )

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")


mri_config = Config()

########################
# DATAMODULE DECLARATION#
########################
datamodule = mri_config.datamodule(config=mri_config)
datamodule.prepare_data()
datamodule.setup()

train_loader = datamodule.train_dataloader()
# mean_train_loader = datamodule.mean_dataloader()
test_loader = datamodule.test_dataloader()

modulator = Modulator(
    dim_in=3, dim_hidden=256, num_layers=mri_config.num_layers
)  # dim_hidden is dim 1 from latent ?

siren = SirenNet(dim_in=3, dim_hidden=mri_config.dim_hidden, dim_out=1, num_layers=4)

model = torch.nn.Sequential(modulator, siren)
optimizer = torch.optim.Adam(params=model.parameters(), lr=mri_config.lr)
model.to("cuda")

# manual training loop
losses = []
for epoch in range(mri_config.epochs):

    # TRAINING LOOP
    for train_batch in train_loader:
        x, y = train_batch
        x = x.to("cuda")
        y = y.to("cuda")

        # x to hash
        mod_lat = modulator(x)
        y_pred = siren(x, mod_lat)
        # hash(x) to modulator
        # x and mod to siren

        loss = F.mse_loss(y_pred, y)
        print(f"epoch: {epoch}")
        print("train loss: ", loss.item())
        losses.append(loss.detach().cpu().numpy())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

x = torch.linspace(-1, 1, 256)
y = torch.linspace(-1, 1, 256)
z = torch.linspace(-1, 1, 256)

mgrid = torch.stack(torch.meshgrid((x, y, z), indexing="ij"), dim=-1)
x_full = TensorDataset(
    torch.reshape(mgrid, (-1, 3)), torch.ones(len(torch.reshape(mgrid, (-1, 3))))
)
loader = DataLoader(x_full, batch_size=mri_config.batch_size, shuffle=False)

pred = torch.zeros(1, 1)

for batch in loader:
    x, y = batch

    x = x.to("cuda")

    mod_lat = modulator(x)
    pred = torch.cat((pred, siren(x, mod_lat).detach().cpu()))

pred = pred[1:, :].numpy()

pred = pred.reshape(256, 256, 256)

nib.save(nib.Nifti1Image(pred, np.eye(4)), "out_modsiren_highres.nii.gz")

plt.plot(range(len(losses)), losses)
plt.savefig("losses_modsiren.png")

# modulator = Modulator(dim_in=encoding_config['n_levels'] * encoding_config['n_features_per_level'], dim_hidden=dim_hidden, num_layers=num_layers)

# net=SirenNet(dim_in=3, dim_hidden=dim_hidden, dim_out=1, num_layers=4)

# net(x)
# modulator(lat2)
