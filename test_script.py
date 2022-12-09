import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union

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

with open("hash_config.json") as f:
    enco_config = json.load(f)


@dataclass
class Config:
    checkpoint_path = ""
    batch_size: int = 16777216 // 50  # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 200
    num_workers: int = os.cpu_count()
    # num_workers:int = 0
    device = [0] if torch.cuda.is_available() else []
    # device = []
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
    model_cls: pl.LightningModule = HashSirenNet

    comment: str = ""

    # output
    output_path: str = "results_hash/"
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path)
    experiment_number: int = 0 if len(os.listdir(output_path)) == 0 else len(
        os.listdir(output_path)
    )

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")

    # TODO: prototype taking into account not typed attributes, loop via dir, but attr value access complicated
    # def export_to_txt(self, file_path: str = '') -> None:
    #     with open(file_path + 'config.txt', 'w') as f:
    #         for attr in dir(self):
    #             if not attr.startswith("__"):
    #                 f.write(attr + ' : ' + str(self.__dict__[key]) + '\n')


config = Config()

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
model.train()

trainer = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=config.accumulate_grad_batches,
)
# trainer = pl.Trainer(gpus=config.device, max_epochs=config.epochs)
trainer.fit(model, train_loader)
model.eval()

image = nib.load(config.image_path)
data = image.get_fdata()
if config.dim_in == 2:
    data = data[:, :, int(data.shape[2] / 2)]
pred = torch.concat(trainer.predict(model, test_loader))

if config.dim_in == 3:
    output = pred.cpu().detach().numpy().reshape(data.shape)
    nib.save(
        nib.Nifti1Image(output, affine=np.eye(4)), filepath + "training_result.nii.gz"
    )
    ground_truth = nib.load(config.image_path).get_fdata()
    ground_truth = (ground_truth - np.min(ground_truth)) / np.max(
        ground_truth
    ) - np.min(ground_truth)
    nib.save(
        nib.Nifti1Image(nib.load(config.image_path).get_fdata(), affine=np.eye(4)),
        filepath + "ground_truth.nii.gz",
    )
if config.dim_in == 2:
    output = pred.cpu().detach().numpy().reshape((data.shape[0], data.shape[1]))
    fig, axes = plt.subplots(1, 2)
    diff = data - output
    axes[0].imshow(output)
    axes[1].imshow(data)
    fig.suptitle("Standard training")
    axes[0].set_title("Prediction")
    axes[1].set_title("Ground truth")
    plt.savefig(filepath + "training_result_standart.png")
    plt.clf()

    plt.imshow(diff)
    plt.savefig(filepath + "difference.png")

config.export_to_txt(file_path=filepath)
