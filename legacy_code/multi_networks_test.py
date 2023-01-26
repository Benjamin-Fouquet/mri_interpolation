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
import torchvision
from types import MappingProxyType

epochs = 1

dataset = torchvision.datasets.MNIST(
            root="mnt/Data/", download=False
        )

class BabyEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        layers = []
        layers.append(torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,padding=1))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3, padding=1))
        layers.append(torch.nn.ReLU())

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class BabyDecoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        layers = []
        layers.append(torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,padding=1))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,padding=1))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3,padding=1))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3,padding=1))
        layers.append(torch.nn.ReLU())

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


encoders = []
for _ in range(10):
    encoders.append(BabyEncoder())

decoder = BabyDecoder()

model = torch.nn.ModuleList((*encoders, decoder))

optimizer = torch.optim.Adam(params=model.parameters())

for epoch in range(epochs):
    for batch in dataset:
        x, y = batch
        x = torchvision.transforms.PILToTensor()(x)
        x = x / torch.max(x)
        lat = encoders[(y - 1)](x)
        y_pred = decoder(lat)
        optimizer.zero_grad()
        loss = torch.nn.MSELoss()(x, y_pred)
        loss.backward()
        optimizer.step()


