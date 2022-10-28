'''
Training loops and outputs

TODO:
-In PsfSirenNet, check that the repeating x and psf match when you add them
-In __init__, add a 1D conv layer with kernel 125 strid 125 as last layer, requiresgrad=False, weights set as psf weights
'''

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Union, Dict
import functorch

# import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
# import functorch
from torch.autograd import Variable
from torchsummary import summary

import math
from einops import rearrange
import pytorch_lightning as pl
import torchvision
import os
from dataclasses import dataclass, field
import sys
import argparse
import copy
import config as cg
import models
import datamodules
import optimizers
import nibabel as nib
from datamodules import MNISTDataModule, MriDataModule

@dataclass
class Config:
    checkpoint_path = ''
    batch_size: int = 2500 #28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 10
    num_workers: int = os.cpu_count()
    # num_workers:int = 0
    device = [0] if torch.cuda.is_available() else []
    # device = []
    image_path:str = 'data/t2_111.nii.gz'
    image_shape = nib.load(image_path).shape
    coordinates_spacing: np.array = np.array((2 / image_shape[0], 2 / image_shape[1], 2 / image_shape[2]))

    #Network parameters
    dim_in: int = 3
    dim_hidden: int = 512
    dim_out:int = 1
    num_layers:int = 5
    w0: float = 30.0
    w0_initial:float = 30.0
    use_bias: bool = True
    final_activation = None
    lr: float = 1e-4 #G requires training with a custom lr, usually lr * 0.1
    datamodule: pl.LightningDataModule = MriDataModule

    comment: str = ''

    #output
    output_path:str = 'results_siren/'
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path)
    experiment_number:int = 0 if len(os.listdir(output_path)) == 0 else len(os.listdir(output_path))

    def export_to_txt(self, file_path: str = '') -> None:
        with open(file_path + 'config.txt', 'w') as f:
            for key in self.__dict__:
                f.write(str(key) + ' : ' + str(self.__dict__[key]) + '\n')

config = Config()

#Correct ouput_path
filepath = config.output_path + str(config.experiment_number) + '/'
if os.path.isdir(filepath) is False:
    os.mkdir(filepath)

###################
#MODEL DECLARATION#
###################
model = models.PsfSirenNet(
    dim_in=config.dim_in,
    dim_hidden=config.dim_hidden,
    dim_out=config.dim_out,
    num_layers=config.num_layers,
    w0=config.w0,
    w0_initial=config.w0_initial,
    use_bias=config.use_bias,
    final_activation=config.final_activation,
    lr=config.lr,
    coordinates_spacing=config.coordinates_spacing
)
########################
#DATAMODULE DECLARATION#
########################
datamodule = config.datamodule(config=config)
datamodule.prepare_data()
datamodule.setup()

train_loader = datamodule.train_dataloader()
# mean_train_loader = datamodule.mean_dataloader()
test_loader = datamodule.test_dataloader()

###################
#PSF TRAINING LOOP#
###################
model.train()
opt = torch.optim.Adam(model.parameters(), lr=config.lr)
trainer = pl.Trainer(gpus=config.device, max_epochs=config.epochs)
trainer.fit(model, train_loader)
model.eval()

# trainer = pl.Trainer(gpus=config.device, max_epochs=config.epochs, accumulate_grad_batches=20)
trainer = pl.Trainer(gpus=config.device, max_epochs=config.epochs)

image = nib.load(config.image_path)
data = image.get_fdata()
if config.dim_in == 2:
    data = data[:,:,int(data.shape[2] / 2)]
pred = torch.concat(trainer.predict(model, test_loader))

if config.dim_in == 3:
    output = pred.cpu().detach().numpy().reshape(data.shape)
    nib.save(nib.Nifti1Image(output, affine=np.eye(4)), filepath + 'training_result.nii.gz')
    nib.save(nib.Nifti1Image(nib.load(config.image_path).get_fdata(), affine=np.eye(4)), filepath + 'ground_truth.nii.gz')
if config.dim_in == 2:
    output = pred.cpu().detach().numpy().reshape((data.shape[0], data.shape[1]))
    fig, axes = plt.subplots(1, 2)
    diff =  data - output
    axes[0].imshow(output)
    axes[1].imshow(data)
    fig.suptitle('Standard training')
    axes[0].set_title('Prediction')
    axes[1].set_title('Ground truth')
    plt.savefig(filepath + 'training_result_standart.png')
    plt.clf()
    
    plt.imshow(diff)
    plt.savefig(filepath + 'difference.png')



