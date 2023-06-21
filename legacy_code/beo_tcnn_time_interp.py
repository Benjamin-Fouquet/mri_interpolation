#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import argparse
import tinycudann as tcnn
import os


class HashMLP(pl.LightningModule):
    def __init__(self, config, dim_in=3, dim_out=1, dim_t=2):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_t = dim_t

        self.encodings = nn.ModuleList(
            [
                tcnn.Encoding(n_input_dims=dim_in, encoding_config=config["encoding"])
                for t in range(dim_t)
            ]
        )
        self.mlp = tcnn.Network(
            n_input_dims=self.encodings[0].n_output_dims,
            n_output_dims=dim_out,
            network_config=config["network"],
        )

    def forward(self, x):
        x_time = x[:, 0].reshape(-1, 1)  # time is first column
        x_pos = x[:, 1:4]  # x,y,z position are 2 to 4th colums
        t = torch.linspace(0, 1, self.dim_t, device=self.device) #timepoints tensor giving one t value for each "frame" to yield

        # compute linear weights
        w = (self.dim_t - 1) * torch.relu(1 / (self.dim_t - 1) - torch.abs(x_time - t)) 

        # weighted encoding
        w_enc = torch.zeros(
            x.shape[0], self.encodings[0].n_output_dims, device=self.device
        )
        for t in range(self.dim_t):
            w_enc += self.encodings[t](x_pos) * w[:, t][:, None]

        return self.mlp(w_enc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)

        loss = F.mse_loss(z, y)

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Beo TCNN")
#     parser.add_argument(
#         "-i",
#         "--input",
#         help="Multiple input images (nifti)",
#         action="append",
#         type=str,
#         required=True,
#     )
#     parser.add_argument(
#         "-o", "--output", help="Output image (nifti)", type=str, required=True
#     )
#     parser.add_argument(
#         "-m",
#         "--model",
#         help="Pytorch lightning (ckpt file) trained model",
#         type=str,
#         required=False,
#     )
#     parser.add_argument(
#         "-b", "--batch_size", help="Batch size", type=int, required=False, default=4096
#     )
#     parser.add_argument(
#         "-e", "--epochs", help="Number of epochs", type=int, required=False, default=10
#     )
#     parser.add_argument(
#         "-n",
#         "--neurons",
#         help="Number of neurons in MLP layers",
#         type=int,
#         required=False,
#         default=128,
#     )
#     parser.add_argument(
#         "-l",
#         "--layers",
#         help="Number of layers in MLP",
#         type=int,
#         required=False,
#         default=2,
#     )
#     parser.add_argument(
#         "-f",
#         "--features",
#         help="Number of features per level (hash grid)",
#         type=int,
#         required=False,
#         default=2,
#     )
#     parser.add_argument(
#         "--levels",
#         help="Number of levels (hash grid)",
#         type=int,
#         required=False,
#         default=8,
#     )
#     parser.add_argument(
#         "--log2_hashmap_size",
#         help="Log2 hashmap size (hash grid)",
#         type=int,
#         required=False,
#         default=15,
#     )  # 15:nvidia, 19: nesvor
#     parser.add_argument(
#         "--base", help="Base resolution", type=int, required=False, default=16
#     )
#     parser.add_argument(
#         "--n_encodings",
#         help="Number of temporal encodings (minimum 2)",
#         type=int,
#         required=False,
#         default=2,
#     )

#     args = parser.parse_args()

image_list = ['data/equinus_frames/frame0.nii.gz', 'data/equinus_frames/frame2.nii.gz', 'data/equinus_frames/frame4.nii.gz']
# image_list = [f'data/equinus_frames/frame{i}.nii.gz' for i in range(15)]
n_images = len(image_list)
output_file = 'prediction.nii.gz'

num_epochs = 10
batch_size = 4096
num_workers = os.cpu_count()

#enco_paramters
n_encodings = len(image_list)
n_t_outputs = 5
base_resolution = 32
n_levels = 8

# Read first image
print("Reading : " + image_list[0])
image = nib.load(image_list[0])
data = image.get_fdata()

# Create grid
dim = 4
nx = data.shape[0]
ny = data.shape[1]
nz = data.shape[2]
nt = n_images
nmax = np.max([nx, ny, nz])

x = torch.linspace(0, 1, steps=nx)
y = torch.linspace(0, 1, steps=ny)
z = torch.linspace(0, 1, steps=nz)
t = torch.linspace(0, 1, steps=nt)
print(t)

mgrid = torch.stack(torch.meshgrid(t, x, y, z, indexing="ij"), dim=-1)

# Convert to X=(x,y,z) and Y=intensity
X = torch.Tensor(mgrid.reshape(-1, dim))
Y = torch.Tensor(data.flatten())

# Normalize intensities between [-1,1]
Y = (Y - torch.min(Y)) / (torch.max(Y) - torch.min(Y)) * 2 - 1
Y = torch.reshape(Y, (-1, 1))

# Add other images
for i in range(1, n_images):
    print("Reading : " + image_list[i])
    image = nib.load(image_list[i])
    data = image.get_fdata()
    Ytmp = torch.Tensor(data.flatten())
    Ytmp = (Ytmp - torch.min(Ytmp)) / (torch.max(Ytmp) - torch.min(Ytmp)) * 2 - 1
    Ytmp = torch.reshape(Ytmp, (-1, 1))
    Y = torch.concat((Y, Ytmp), dim=0)

print(X.shape)
print(Y.shape)

# Pytorch dataloader
dataset = torch.utils.data.TensorDataset(X, Y)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

# Training
b = np.exp((np.log(nmax) - np.log(base_resolution)) / (n_levels - 1))

# https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
config = {
    "encoding": {
        "otype": "HashGrid",
        "n_levels": n_levels,
        "n_features_per_level": 2,
        "log2_hashmap_size": 15,
        "base_resolution": base_resolution,
        "per_level_scale": b,  # 1.3819#1.5
    },
    "network": {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 128,
        "n_hidden_layers": 2,
    },
}

net = HashMLP(config=config, dim_in=3, dim_out=1, dim_t=n_encodings)
net = HashMLP.load_from_checkpoint('lightning_logs/version_13/checkpoints/epoch=9-step=5450.ckpt', config=config, dim_in=3, dim_out=1, dim_t=n_encodings)
trainer = pl.Trainer(gpus=[0] if torch.cuda.is_available() else 0, max_epochs=num_epochs, precision=16)

# net = torch.compile(net)   #Not working for old GPU like Titan
trainer.fit(net, loader)

filepath = net.logger.log_dir + '/'

test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
)  # remove shuffling
yhat = torch.concat(trainer.predict(net, test_loader))

print(yhat.shape)
output = np.float32(
    np.moveaxis(yhat.cpu().detach().numpy().reshape((nt, nx, ny, nz)), 0, 3)
)
print(output.shape)
nib.save(nib.Nifti1Image(output, image.affine), filepath + output_file)

# Prediction at a specific time
t = torch.linspace(0, 1, steps=n_t_outputs)
# t = torch.Tensor([0,0.1,0.2,0.25])
mgrid = torch.stack(torch.meshgrid(t, x, y, z, indexing="ij"), dim=-1)
X = torch.Tensor(mgrid.reshape(-1, dim))
Y = torch.zeros((X.shape[0], 1))  # dummy values
pred_dataset = torch.utils.data.TensorDataset(X, Y)
pred_loader = torch.utils.data.DataLoader(
    pred_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
)
pred = torch.concat(trainer.predict(net, pred_loader))
data4d = np.moveaxis(
    pred.cpu().detach().numpy().reshape((t.shape[0], nx, ny, nz)), 0, 3
)
nib.save(nib.Nifti1Image(np.float32(data4d), image.affine), filepath + "pred_spe.nii.gz")

def export_to_txt(dict, file_path: str = "") -> None:
    with open(file_path + "config.txt", "a+") as f:
        for key in dict:
            f.write(str(key) + " : " + str(dict[key]) + "\n")

