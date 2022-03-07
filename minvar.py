'''
Tentative minimal implementation of Varnet and co
TODO:
-remove need for setup after instantation
'''
from pyexpat import model
from sys import path_importer_cache
import torch
from torch import nn
from dataclasses import dataclass
from mri_dataloading import MriDataModule
import pytorch_lightning as pl

#Dataclass for hyperparameters
@dataclass
class Hyperparameters:
    phi_channels: list
    phi_lr: float

#Class phi = CNN #needs to be pretrained, so you have to test if you can load it from prior trianing data
class Phi_CNN(nn.Module):
    def __init__(
        self,
        num_channels=(128, 128),
        kernel_size=3,
        activation_func="ReLU",
        lr = 0.001,
        *args,
        **kwargs,
                ):
        super().__init__(),
        self.num_channels = num_channels,
        self.kernel_size = 3
        self.activation_func = activation_func
        self.lr = lr
        layers = []
        for idx in range(len(num_channels)):
            in_channels = num_channels[idx - 1] if idx > 0 else 1
            out_channels = num_channels[idx]
            layer = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            )

            layers.append(layer)
            if self.activation_func == "Tanh":
                layers.append(nn.Tanh())
            if self.activation_func == "ReLU":
                layers.append(nn.ReLU())
            if self.activation_func == "Sig":
                layers.append(nn.Sigmoid())
        last_layer = nn.Conv3d(
            in_channels=num_channels[-1],
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )
        layers.append(last_layer)
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    

#Runner
class MiniRunner:
    def __init__(self, datamodule: pl.LightningDataModule, hyperparameters: Hyperparameters, gpu=2) -> None:
        self.datamodule = datamodule
        self.datamodule.setup()
        self.hyperparameters = hyperparameters
        self.gpu = gpu
        self.model = model
        self.phi = Phi_CNN(num_channels=hyperparameters.phi_channels, lr=hyperparameters.phi_lr) #you may need the checkpoint

    def setup(self):
        pass

    def train(self):
        pass

    def val(self):
        pass

    def test(self):
        pass

phi_channels = [128, 128]
phi_lr = 0.001
mri_datamodule = MriDataModule()
mri_datamodule.setup()
xp_hyperparameters=Hyperparameters(phi_channels=phi_channels, phi_lr=phi_lr) #do you really need parameters when pre trained model ?

runner = MiniRunner(datamodule=mri_datamodule, hyperparameters=xp_hyperparameters)
runner.setup()
