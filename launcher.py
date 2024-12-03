"""
Launcher for trainings using datamodules and models
"""
import argparse
import glob
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Tuple, Union

import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing as proc
# import pytorch_lightning as pl
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from skimage import metrics
from torch import nn
from torch.nn import functional as F
# import functorch
from torchsummary import summary

from config import base
from datamodules import MriDataModule
import models

torch.manual_seed(1337)

args = None

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
    parser.add_argument(
        "--enco_config_path",
        help="path for tinycuda encoding config",
        type=str,
        required=False,
    )
    args = parser.parse_args()


def export_to_txt(dict: dict, file_path: str = "") -> None:
    """
    Helper function to export dictionary to text file
    """
    with open(file_path + "config.txt", "a+") as f:
        for key in dict:
            f.write(str(key) + " : " + str(dict[key]) + "\n")


config = base.HashConfig()

with open("config/hash_config.json") as f:
    config.enco_config = json.load(f)

# parsed argument -> config
if args is not None:
    for key in args.__dict__:
        if args.__dict__[key] is not None:
            config.__dict__[key] = args.__dict__[key]

# # correct for model class #could use getattr() here
# if args.model_class is not None:
#     try:
#         config.model_cls = getattr(models, args.model_class)
#     except:
#         print('model class not recognized, exiting')  
#         sys.exit()  

training_start = time.time()
####################
# MODEL DECLARATION#
####################

#Not all arguments are used for all models, ideally the next implementation would make use of specialized library (hydra) or create a class to decompose the config and feed it to the model class

if config.checkpoint_path:
    model = config.model_cls.load_from_checkpoint(
        config.checkpoint_path,
        dim_in=config.dim_in,
        dim_hidden=config.dim_hidden,
        dim_out=config.dim_out,
        n_layers=config.n_layers,
        encoder_type=config.encoder_type,
        n_levels=config.n_levels,
        n_features_per_level=config.n_features_per_level,
        log2_hashmap_size=config.log2_hashmap_size,
        base_resolution=config.base_resolution,
        finest_resolution=config.finest_resolution,
        per_level_scale=config.per_level_scale,
        interpolation=config.interpolation,
        w0=config.w0,
        w0_initial=config.w0_initial,
        use_bias=config.use_bias,
        final_activation=config.final_activation,
        lr=config.lr,
    )
    
else:

    model = config.model_cls(
        dim_in=config.dim_in,
        dim_hidden=config.dim_hidden,
        dim_out=config.dim_out,
        n_layers=config.n_layers,
        encoder_type=config.encoder_type,
        n_levels=config.n_levels,
        n_features_per_level=config.n_features_per_level,
        log2_hashmap_size=config.log2_hashmap_size,
        base_resolution=config.base_resolution,
        finest_resolution=config.finest_resolution,
        per_level_scale=config.per_level_scale,
        interpolation=config.interpolation,
        w0=config.w0,
        w0_initial=config.w0_initial,
        use_bias=config.use_bias,
        final_activation=config.final_activation,
        lr=config.lr,
    )

#########################
# DATAMODULE DECLARATION#
#########################

datamodule = config.datamodule(config=config)
datamodule.prepare_data()
datamodule.setup()

train_loader = datamodule.train_dataloader()
test_loader = datamodule.test_dataloader()

################
# TRAINING LOOP#
################

trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    max_epochs=config.epochs,
    accumulate_grad_batches=dict(config.accumulate_grad_batches)
    if config.accumulate_grad_batches
    else None,
    precision=32,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)
trainer.fit(model, train_loader)

training_stop = time.time()

#######################
#PREDICTION AND OUTPUT#
#######################

filepath = model.logger.log_dir + '\\'
# filepath = model.logger.log_dir + '/'

config.log = str(model.logger.version)

# create a prediction
pred = torch.concat(trainer.predict(model, test_loader))

# im = pred.reshape(config.image_shape)
im = pred.reshape(config.image_shape)
im = im.detach().cpu().numpy()
im = np.array(im, dtype=np.float32)
if len(im.shape) == 2:
    plt.imshow(im.T)
    plt.savefig(filepath + "pred.png")
else:
    nib.save(nib.Nifti1Image(im, affine=np.eye(4)), filepath + "pred.nii.gz")

for shape in config.interp_shapes:
    # dense grid
    Y_interp = torch.zeros((np.prod(shape), 1))

    axes = []
    for s in config.interp_shape:
        axes.append(torch.linspace(0, 1, s))

    mgrid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)

    coords = torch.FloatTensor(mgrid)
    X_interp = coords.reshape(len(Y_interp), config.dim_in)

    interp_dataset = torch.utils.data.TensorDataset(X_interp, Y_interp)
    interp_loader = torch.utils.data.DataLoader(
        interp_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # create an interpolation
    interp = torch.concat(trainer.predict(model, interp_loader))

    interp_im = interp.reshape(shape)

    interp_im = interp_im.detach().cpu().numpy()
    interp_im = np.array(interp_im, dtype=np.float32)
    nib.save(
        nib.Nifti1Image(interp_im, affine=np.eye(4)),
        filepath + f"interpolation{shape}.nii.gz",
    )

config.export_to_txt(file_path=filepath)





