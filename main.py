'''
Training loops and outputs
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

# from utils import psf_kernel, apply_psf, expend_x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inner_loop_it', help='Inner loop iterations', type=int, required=False)
    parser.add_argument('--outer_loop_it', help='Outer loop iterations', type=int, required=False)
    parser.add_argument('--epochs', help='Number of epochs', type=int, required=False)
    parser.add_argument('--train_target', help='train digit', type=int, required=False)
    parser.add_argument('--test_target', help='test digit', type=int, required=False)
    parser.add_argument('--opt_type', help='optimizer model type', type=str, required=False)

    args = parser.parse_args()

config = cg.Config()

#parsed argument -> config
for key in args.__dict__:
    if args.__dict__[key] is not None:
        config.__dict__[key] = args.__dict__[key]

#Correct ouput_path
filepath = config.output_path + str(config.experiment_number) + '/'
if os.path.isdir(filepath) is False:
    os.mkdir(filepath)

if config.fixed_seed:
    torch.random.manual_seed(0)

###################
#MODEL DECLARATION#
###################

# #TODO: switch to config declaration?
model = models.SirenNet(
    dim_in=config.dim_in,
    dim_hidden=config.dim_hidden,
    dim_out=config.dim_out,
    num_layers=config.num_layers,
    w0=config.w0,
    w0_initial=config.w0_initial,
    use_bias=config.use_bias,
    final_activation=config.final_activation,
)

# model = models.FourrierNet(
#     dim_in=config.dim_in,
#     dim_hidden=config.dim_hidden,
#     dim_out=config.dim_out,
#     num_layers=config.num_layers,
#     w0_initial=config.w0_initial,
#     use_bias=config.use_bias,
#     final_activation=config.final_activation,
# )

model_func, theta_init = functorch.make_functional(model)

# datamodule = datamodules.MNISTDataModule(config=config)
datamodule = datamodules.MriDataModule(config=config)
datamodule.prepare_data()
datamodule.setup()

train_loader = datamodule.train_dataloader()
mean_train_loader = datamodule.mean_dataloader()

init_data = nib.load('data/mean.nii.gz')

#################
#INITIALIZATIONS#
#################
# psf = psf_kernel()
# model.set_parameters(theta_init)
# model.train()
# model_losses_adam_opt_mean = []
# opt = torch.optim.Adam(model.parameters(), lr=config.lr)
# for _ in range(config.epochs):
#     x, y = next(iter(mean_train_loader))
#     # #expend x with psf and reflatten
#     # x = expend_x(x, init_data)
#     # #predict
#     y_pred = model(x)
#     # #sum prediction over PSF
#     # y_pred = psf_sum(y_pred, psf)
#     # #compare with y
#     loss = F.mse_loss(y_pred, y)


#     # if config.apply_psf:
#     #     y_pred = apply_psf(tensor=y_pred, kernel=psf, image_shape=(290, 290))
#     # loss = F.mse_loss(y_pred, y)    
#     opt.zero_grad()
#     loss.backward()
#     opt.step()

#     print(f'Loss: {loss.data}')
#     model_losses_adam_opt_mean.append(loss.detach().numpy())

model_func, theta_mean = functorch.make_functional(model)

########################
#STANDARD TRAINING LOOP#
########################
training_data = nib.load(config.image_path)
model.set_parameters(theta_mean)
model.train()
model_losses_adam_opt = []
opt = torch.optim.Adam(model.parameters(), lr=config.lr)
trainer = pl.Trainer(gpus=[0], max_epochs=config.epochs)
trainer.fit(model, train_loader)

# for _ in range(config.epochs):
#     x, y = next(iter(train_loader))
#     y_pred = model(x)
#     # if config.apply_psf:
#     #     y_pred = apply_psf(tensor=y_pred, kernel=psf, image_shape=(260, 311))
#     loss = F.mse_loss(y_pred, y)

#     opt.zero_grad()
#     loss.backward()
#     opt.step()

#     print(f'Loss: {loss.data}')
#     model_losses_adam_opt.append(loss.detach().numpy())

#TODO: correct prediction for MRI
#squeeze?

if isinstance(datamodule, datamodules.MNISTDataModule):
    pred = model(x)
    image = pred.reshape((28, 28))
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image.detach().numpy())
    axes[1].imshow(y.detach().numpy().reshape(28, 28))
    fig.suptitle('Standard training')
    axes[0].set_title('Prediction')
    axes[1].set_title('Ground truth')
    plt.savefig(filepath + 'training_result_standart.png')
    plt.clf()

if isinstance(datamodule, datamodules.MriDataModule):
    trainer = pl.Trainer(gpus=config.device, max_epochs=config.epochs)
    image = nib.load(config.image_path)
    data = image.get_fdata()
    if config.dim_in == 2:
        data = data[:,:,int(data.shape[2] / 2)]
    pred = torch.concat(trainer.predict(model, train_loader))

    if config.dim_in == 3:
        output = pred.cpu().detach().numpy().reshape(data.shape)
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

mean_loader = datamodule.mean_dataloader()
x, y = next(iter(mean_loader))
y = y.detach().numpy().reshape(290, 290)
plt.imshow(y)
plt.show()
plt.clf()

plt.plot(range(len(model_losses_adam_opt)), model_losses_adam_opt)
plt.savefig(filepath + 'Losses_sdt_opt.png')

trainer.save_checkpoint(filepath + 'checkpoint.ckpt') 
    