'''
WIP hsearch script, made with optuna
'''

import torch

import optuna

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
from models import HashSirenNet, ModulatedSirenNet, PsfSirenNet, SirenNet
# import functorch
from torchsummary import summary
from config.base import BaseConfig
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch size", type=int, required=False)
    parser.add_argument("--epochs", help="Number of epochs", type=int, required=False)
    parser.add_argument(
        "--accumulate_grad_batches",
        help="number of batches accumulated per gradient descent step",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--n_sample",
        help="number of points for psf in x, y, z",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--model_class", help="Modele class selection", type=str, required=False
    )

    args = parser.parse_args()

config = BaseConfig()

with open(config.hashconfig_path) as f:
    enco_config = json.load(f)

# parsed argument -> config
for key in args.__dict__:
    if args.__dict__[key] is not None:
        config.__dict__[key] = args.__dict__[key]

# correct for model class
if args.model_class is not None:
    if args.model_class == "PsfSirenNet":
        config.model_cls = PsfSirenNet
    elif args.model_class == "SirenNet":
        config.model_cls = SirenNet
    else:
        print("model class not recognized")
        raise ValueError

# Correct ouput_path
filepath = config.output_path + str(config.experiment_number) + "/"
if os.path.isdir(filepath) is False:
    os.mkdir(filepath)

# 1. Define an objective function to be maximized.
def objective(trial):

    # 2. Suggest values of the hyperparameters using a trial object.
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int(f'n_units_l{i}', 4, 128)
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        in_features = out_features
    layers.append(torch.nn.Linear(in_features, 10))
    layers.append(torch.nn.LogSoftmax(dim=1))
    model = torch.nn.Sequential(*layers).to(torch.device('cpu'))
    ...
    loss = 0
    return loss

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

###################
# MODEL DECLARATION#
###################
model = config.model_cls(
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
    # coordinates_spacing=config.coordinates_spacing,
    # n_sample=config.n_sample
)
########################
# DATAMODULE DECLARATION#
########################
datamodule = config.datamodule(config=config)
datamodule.prepare_data()
datamodule.setup()

train_loader = datamodule.train_dataloader()
# mean_train_loader = datamodule.mean_dataloader()
test_loader = datamodule.test_dataloader()

###############
# TRAINING LOOP#
###############
def objective(trial):
    #trial parameters declaration
    num_layers = trial.suggest_int('num_layers', 3, 6)

    #model declaration
    model = config.model_cls(
    dim_in=config.dim_in,
    dim_hidden=config.dim_hidden,
    dim_out=config.dim_out,
    num_layers=num_layers,
    w0=config.w0,
    w0_initial=config.w0_initial,
    use_bias=config.use_bias,
    final_activation=config.final_activation,
    lr=config.lr,
    config=enco_config,
    # coordinates_spacing=config.coordinates_spacing,
    # n_sample=config.n_sample
    )

    trainer = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=config.accumulate_grad_batches,
)
    trainer.fit(model, train_dataloaders=train_loader)
    loss = model.losses[-1]
    return loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

# model.train()

# trainer = pl.Trainer(
#     gpus=config.device,
#     max_epochs=config.epochs,
#     accumulate_grad_batches=config.accumulate_grad_batches,
# )
# # trainer = pl.Trainer(gpus=config.device, max_epochs=config.epochs)
# trainer.fit(model, train_loader)
# model.eval()

# image = nib.load(config.image_path)
# data = image.get_fdata()
# if config.dim_in == 2:
#     data = data[:, :, int(data.shape[2] / 2)]
# pred = torch.concat(trainer.predict(model, test_loader))

# if config.dim_in == 3:
#     output = pred.cpu().detach().numpy().reshape(data.shape)
#     nib.save(
#         nib.Nifti1Image(output, affine=np.eye(4)), filepath + "training_result.nii.gz"
#     )
#     ground_truth = nib.load(config.image_path).get_fdata()
#     ground_truth = (ground_truth - np.min(ground_truth)) / np.max(
#         ground_truth
#     ) - np.min(ground_truth)
#     nib.save(
#         nib.Nifti1Image(nib.load(config.image_path).get_fdata(), affine=np.eye(4)),
#         filepath + "ground_truth.nii.gz",
#     )
# if config.dim_in == 2:
#     output = pred.cpu().detach().numpy().reshape((data.shape[0], data.shape[1]))
#     fig, axes = plt.subplots(1, 2)
#     diff = data - output
#     axes[0].imshow(output)
#     axes[1].imshow(data)
#     fig.suptitle("Standard training")
#     axes[0].set_title("Prediction")
#     axes[1].set_title("Ground truth")
#     plt.savefig(filepath + "training_result_standart.png")
#     plt.clf()

#     plt.imshow(diff)
#     plt.savefig(filepath + "difference.png")

# config.export_to_txt(file_path=filepath)
