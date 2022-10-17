'''
Datamodules for MRI interpolation. Each module comes with an optional 2D MNIST setup for tests purposes. Not all datamodules are compatible with all networks (implicit representation vs conventional voxel view)
Data options:
-MNIST for tests
-DHCP for real deal
-potentially fabien and/or foetal to be added lated
-Also, right tests for datamodules ?
'''

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Union, Dict, Optional
from torch.utils.data import DataLoader, Dataset
# import torchio as tio
import nibabel as nib

# import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

import math
from einops import rearrange
import pytorch_lightning as pl
import torchvision
import os
from dataclasses import dataclass, field
import sys
import argparse
import copy
import config as cf #TODO: inelegant, replace by hydra at a point
import matplotlib.pyplot as plt
from utils import create_rn_mask

def display_output(batch):
    x, y = batch
    x = x.squeeze()
    y = y.squeeze()
    plt.imshow(y.reshape(28, 28))
    plt.show()
    plt.clf()

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, config=None):
        super().__init__()
        self.config = config

    def prepare_data(self) -> None:
        '''
        Steps to be done on 1 GPU, like datafetch
        '''
        self.mnist_dataset = torchvision.datasets.MNIST(
        root=self.config.dataset_path, download=False
    )
        return super().prepare_data()

    def setup(self) -> None:
        '''
        steps to be done on multiple GPUs, like datatransforms
        '''
        if self.config.initialization == 'mean':
            #get all digits from train_targets and mean them
            digit_tensor = torch.zeros(28, 28)
            for digit_idx, target in enumerate(self.mnist_dataset.targets):
                if int(target) in self.config.train_target:
                    digit_tensor += torchvision.transforms.ToTensor()(self.mnist_dataset[digit_idx][0]).squeeze()
                    
        #fetch the wanted train digit, 1 digit version
        if self.config.initialization == 'single':
            for digit_idx, target in enumerate(self.mnist_dataset.targets):
                if int(target) in self.config.train_target:
                    digit_tensor = torchvision.transforms.ToTensor()(self.mnist_dataset[digit_idx][0]).squeeze()
                    break

        if self.config.initialization == 'random':
            targets_list = []
            for digit_idx, target in enumerate(self.mnist_dataset.targets):
                if int(target) in self.config.train_target:
                    targets_list.append(self.mnist_dataset[digit_idx][0])
            digit_tensor = torch.empty(0)
            for idx in range(len(targets_list)):
                target_tensor = torchvision.transforms.ToTensor()(targets_list[idx]).squeeze()
                flat = torch.Tensor(target_tensor.flatten()).unsqueeze(-1)
                digit_tensor = torch.cat((digit_tensor, flat))

        #normalization
        digit_tensor = digit_tensor * 2 - 1
        digit_shape = [28, 28] #hardcoded MNIST

        x = torch.linspace(-1, 1, digit_shape[0])
        y = torch.linspace(-1, 1, digit_shape[1])
        mgrid = torch.stack(torch.meshgrid(x, y), dim=-1)

        if self.config.initialization == 'random':
            x_flat = torch.Tensor(mgrid.reshape(-1, 2)).repeat(len(targets_list), 1)
            y_flat  = digit_tensor
        else:
            x_flat = torch.Tensor(mgrid.reshape(-1, 2))
            y_flat = torch.Tensor(digit_tensor.flatten()).unsqueeze(-1)

        self.dataset = torch.utils.data.TensorDataset(x_flat, y_flat)

        return None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
    self.dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers
)

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None

class MriImage(Dataset):
    def __init__(self, config, image_path=None, *args, **kwargs):
        super().__init__()
        if image_path:
            image = nib.load(image_path)
        else:
            image = nib.load(config.image_path)
        image = image.get_fdata(dtype=np.float32)    #[64:192, 64:192, 100:164]
        if config.dim_in == 3:
            x = torch.linspace(-1, 1, steps=image.shape[0])
            y = torch.linspace(-1, 1, steps=image.shape[1])
            z = torch.linspace(-1, 1, steps=image.shape[2])
            mgrid = torch.stack(torch.meshgrid(x,y,z), dim=-1)
        if config.dim_in == 2:
            x = torch.linspace(-1, 1, steps=image.shape[0])
            y = torch.linspace(-1, 1, steps=image.shape[1])
            mgrid = torch.stack(torch.meshgrid(x,y), dim=-1)

        #create data tensors
        pixels = torch.FloatTensor(image)
        if config.dim_in == 2:
            pixels = pixels[:,:,int(pixels.shape[2] / 2)]
        pixels = pixels.flatten()
        #normalisation, should be recasted with torch reshape func
        pixels = ((pixels - torch.min(pixels)) / torch.max(pixels)) * 2 - 1
        coords = torch.FloatTensor(mgrid)
        coords = coords.reshape(len(pixels), config.dim_in)
        assert len(coords) == len(pixels)
        self.coords = coords
        self.pixels = pixels.unsqueeze(-1)

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):  
        return self.coords[idx], self.pixels[idx]

class MriDataModule(pl.LightningDataModule):
    '''
    Take ONE mri image and returns coords and pixels, no split on train/val/test for the moment
    '''
    def __init__(
        self,
        config=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.config = config

    def prepare_data(self) -> None:
        self.dataset = MriImage(config=self.config)
        self.mean_dataset = MriImage(config=self.config, image_path='data/mean.nii.gz') #how to set mean without screwing up ?
        self.train_ds = self.dataset
        self.test_ds = self.dataset
        self.val_ds = self.dataset
        
    def setup(self, normalisation: str = 'zero centered'):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.config.batch_size, num_workers=self.config.num_workers, shuffle=True)

    def val_dataloader(self)-> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def test_dataloader(self)-> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def mean_dataloader(self)->DataLoader:
        return DataLoader(self.mean_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
    


# #crude tests, with test_configs
# if __name__ == 'main':

#     @dataclass
#     class TestConfig:
#         initialization: str = 'single'
#         dataset_path: str = '/home/benjamin/Documents/Datasets'
#         train_target = [2]
#         test_target = [2]
#         batch_size = 784
#         num_workers = os.cpu_count()

#     test_config = TestConfig()
#         #check initialization
#     datamodule = MNISTDataModule(config=test_config)
#     datamodule.prepare_data()
#     datamodule.setup()

#     dataloader = datamodule.train_dataloader()

#     display_output(next(iter(dataloader)))

#     test_config = TestConfig(initialization='mean')
#     datamodule = MNISTDataModule(config=test_config)
#     datamodule.prepare_data()
#     datamodule.setup()

#     dataloader = datamodule.train_dataloader()

#     display_output(next(iter(dataloader)))

#     test_config = TestConfig(initialization='random')
#     datamodule = MNISTDataModule(config=test_config)
#     datamodule.prepare_data()
#     datamodule.setup()

#     dataloader = datamodule.train_dataloader()

#     display_output(next(iter(dataloader)))

#     it = iter(dataloader)
#     for i in range(4):
#         display_output(next(it))