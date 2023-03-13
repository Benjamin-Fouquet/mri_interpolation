'''
Minimalist script for VAE testing with MNIST /!\ diffusion instead
'''
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
import torchvision

torch.manual_seed(1337)

@dataclass
class MNISTConfig:
    batch_size: int = 784  # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST
    inner_loop_it: int = 5
    outer_loop_it: int = 10
    epochs: int = 1
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    fixed_seed: bool = True
    # dataset_path: str = '/home/benjamin/Documents/Datasets' #for MNIST
    dataset_path: str = "mnt/Data/"
    # image_path: str = '/home/benjamin/Documents/Datasets/HCP/100307_T2.nii.gz'
    image_path: str = "data/t2_256cube.nii.gz"
    train_target: tuple = (2,)
    test_target: tuple = (7,)
    initialization: str = "single"
    apply_psf: bool = False
    hashconfig_path: str = 'hash_config.json'

    # Network parameters
    dim_in: int = 2
    dim_hidden: int = 256
    dim_out: int = 1
    num_layers: int = 5
    w0: float = 1.0
    w0_initial: float = 30.0
    use_bias: bool = True
    final_activation = None
    lr: float = 1e-3  # G requires training with a custom lr, usually lr * 0.1
    opt_type: str = "LSTM"
    conv_channels: tuple = (8, 8, 8,)
    datamodule: pl.LightningDataModule = MNISTDataModule
    accum: MappingProxyType = MappingProxyType({200: 3, 400: 4})

    # output
    output_path: str = "results_fourrier/"
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path)
    experiment_number: int = 0 if len(os.listdir(output_path)) == 0 else len(
        os.listdir(output_path)
    )

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")


class MNISTDataModule(pl.LightningDataModule):
    '''
    Datamodule for implicit representation training based on MNIST dataset
    loader returns coordinates (-1, 1) as x and intensity as y
    '''
    def __init__(self, config=None):
        super().__init__()
        self.config = config

    def prepare_data(self) -> None:
        """
        Steps to be done on 1 GPU, like datafetch
        """
        self.mnist_dataset = torchvision.datasets.MNIST(
            root=self.config.dataset_path, download=False
        )
        return super().prepare_data()

    def setup(self) -> None:
        """
        steps to be done on multiple GPUs, like datatransforms
        """
        if self.config.initialization == "mean":
            # get all digits from train_targets and mean them
            digit_tensor = torch.zeros(28, 28)
            for digit_idx, target in enumerate(self.mnist_dataset.targets):
                if int(target) in self.config.train_target:
                    digit_tensor += torchvision.transforms.ToTensor()(
                        self.mnist_dataset[digit_idx][0]
                    ).squeeze()

        # fetch the wanted train digit, 1 digit version
        if self.config.initialization == "single":
            for digit_idx, target in enumerate(self.mnist_dataset.targets):
                if int(target) in self.config.train_target:
                    digit_tensor = torchvision.transforms.ToTensor()(
                        self.mnist_dataset[digit_idx][0]
                    ).squeeze()
                    break

        if self.config.initialization == "random":
            targets_list = []
            for digit_idx, target in enumerate(self.mnist_dataset.targets):
                if int(target) in self.config.train_target:
                    targets_list.append(self.mnist_dataset[digit_idx][0])
            digit_tensor = torch.empty(0)
            for idx in range(len(targets_list)):
                target_tensor = torchvision.transforms.ToTensor()(
                    targets_list[idx]
                ).squeeze()
                flat = torch.Tensor(target_tensor.flatten()).unsqueeze(-1)
                digit_tensor = torch.cat((digit_tensor, flat))

        # normalization
        digit_tensor = digit_tensor * 2 - 1
        digit_shape = [28, 28]  # hardcoded MNIST

        x = torch.linspace(-1, 1, digit_shape[0])
        y = torch.linspace(-1, 1, digit_shape[1])
        mgrid = torch.stack(torch.meshgrid(x, y), dim=-1)

        if self.config.initialization == "random":
            x_flat = torch.Tensor(mgrid.reshape(-1, 2)).repeat(len(targets_list), 1)
            y_flat = digit_tensor
        else:
            x_flat = torch.Tensor(mgrid.reshape(-1, 2))
            y_flat = torch.Tensor(digit_tensor.flatten()).unsqueeze(-1)

        self.dataset = torch.utils.data.TensorDataset(x_flat, y_flat)

        return None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None


