import torch
import torch.nn as nn
import pytorch_lightning as pl

data = torch.randn(352, 352, 4)
dim_hidden=352

class MLP(pl.LightningModule):
    def __init__(self, dim_in, dim_hidden, dim_out, n_layers, lr):
        super().__init__()
        self.lr = lr
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_features=dim_in if i == 0 else dim_hidden, out_features=dim_out if i == (n_layers - 1) else dim_hidden))
            layers.append(nn.ReLU())
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)

        loss = F.mse_loss(z, y)
        
        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
model = MLP(dim_in=len(data.shape), dim_hidden=dim_hidden, dim_out=1, n_layers=8, lr=1e-4)


# import vtk
# import argparse
# from scipy.interpolate import griddata, interpn
# from torch import autograd as ag
# import torch

# def nii_2_mesh(filename_nii, filename_stl, label):

#     """
#     Read a nifti file including a binary map of a segmented organ with label id = label. 
#     Convert it to a smoothed mesh of type stl.

#     filename_nii     : Input nifti binary map 
#     filename_stl     : Output mesh name in stl format
#     label            : segmented label id 
#     """

#     # read the file
#     reader = vtk.vtkNIFTIImageReader()
#     reader.SetFileName(filename_nii)
#     reader.Update()
    
#     # apply marching cube surface generation
#     surf = vtk.vtkDiscreteMarchingCubes()
#     surf.SetInputConnection(reader.GetOutputPort())
#     surf.SetValue(0, label) # use surf.GenerateValues function if more than one contour is available in the file	
#     surf.Update()
    
#     #  auto Orient Normals
#     surf_cor = vtk.vtkPolyDataNormals()
#     surf_cor.SetInputConnection(surf.GetOutputPort())
#     surf_cor.ConsistencyOn()
#     surf_cor.AutoOrientNormalsOn()
#     surf_cor.SplittingOff()
#     surf_cor.Update()
    
#     #smoothing the mesh
#     smoother= vtk.vtkWindowedSincPolyDataFilter()
#     if vtk.VTK_MAJOR_VERSION <= 5:
#         smoother.SetInput(surf_cor.GetOutput())
#     else:
#         smoother.SetInputConnection(surf_cor.GetOutputPort())
#     smoother.SetNumberOfIterations(60)
#     smoother.NonManifoldSmoothingOn()
#     #smoother.NormalizeCoordinatesOn() #The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
#     smoother.GenerateErrorScalarsOn()
#     smoother.Update()
     
#     # save the output
#     writer = vtk.vtkSTLWriter()
#     writer.SetInputConnection(smoother.GetOutputPort())
#     writer.SetFileTypeToASCII()
#     writer.SetFileName(filename_stl)
#     writer.Write()

# #extract (x, y, z), how many points ? 

# t_zero = torch.FloatTensor((0, 0, 0))
# t_one = torch.FloatTensor((1, 1, 1))
# t_zero.requires_grad = True
# t_one.requires_grad = True

# delta_t = t_one - t_zero

# ag.grad(outputs=delta_t, inputs=t_one)

# from typing import List, Optional, Union
# from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
# import tinycudann as tcnn 
# import torch
# import pytorch_lightning as pl 
# import torch.nn.functional as F
# import json
# import nibabel as nib 
# from dataclasses import dataclass
# import os
# from types import MappingProxyType
# import numpy as np
# import math
# import rff
# import argparse
# from torch.utils.tensorboard import SummaryWriter
# from math import pi
# import torch.utils.data
# import matplotlib.pyplot as plt

# torch.manual_seed(1337)

# @dataclass
# class BaseConfig:
#     checkpoint_path = None #'lightning_logs/version_25/checkpoints/epoch=49-step=11200.ckpt'
#     log: str = None
#     # image_path: str = '/mnt/Data/FetalAtlas/template_T2.nii.gz'
#     image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
#     image_shape = nib.load(image_path).shape
#     batch_size: int = 30000 #~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
#     epochs: int = 50
#     num_workers: int = os.cpu_count()
#     device = [0] if torch.cuda.is_available() else []
#     accumulate_grad_batches: MappingProxyType = None 
#     # Network parameters
#     n_encoders: int = 9 
#     n_frequencies: int = 4  #for classic, n_out = 2 * n_freq. For tcnn, n_out = 2 * n_freq * dim_in
#     sigma: float = 2.0
#     dim_in: int = len(image_shape)
#     dim_hidden: int = 128
#     dim_out: int = 1
#     num_layers: int = 4
#     skip_connections: tuple = () #(5, 11,)
#     lr: float = 1e-3  # G requires training with a custom lr, usually lr * 0.1 
#     interp_factor: int = 2

#     def export_to_txt(self, file_path: str = "") -> None:
#         with open(file_path + "config.txt", "w") as f:
#             for key in self.__dict__:
#                 f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--batch_size", help="batch size", type=int, required=False)
#     parser.add_argument("--epochs", help="Number of epochs", type=int, required=False)
#     parser.add_argument("--image_path", help="path of image", type=str, required=False)
#     parser.add_argument("--encoder_type", help="tcnn or classic", type=str, required=False)
#     parser.add_argument("--n_frequencies", help="number of encoding frequencies", type=int, required=False)
#     parser.add_argument("--n_frequencies_t", help="number of encoding frequencies for time", type=int, required=False)
#     args = parser.parse_args()

# def export_to_txt(dict: dict, file_path: str = "") -> None:
#     '''
#     Helper function to export dictionary to text file
#     '''
#     with open(file_path + "config.txt", "a+") as f:
#         for key in dict:
#             f.write(str(key) + " : " + str(dict[key]) + "\n")

# config = BaseConfig()

# # parsed argument -> config
# for key in args.__dict__:
#     if args.__dict__[key] is not None:
#         config.__dict__[key] = args.__dict__[key]


# #freq encoding tiny cuda
# class FreqMLP(pl.LightningModule):
#     '''
#     Lightning module for HashMLP. 
#     '''
#     def __init__(
#         self,
#         dim_in,
#         dim_hidden,
#         dim_out,
#         n_layers,
#         skip_connections,
#         n_encoders,
#         n_frequencies,
#         sigma,
#         lr,
#         *args,
#         **kwargs
#     ):
#         super().__init__()
#         self.dim_in = dim_in
#         self.dim_hidden = dim_hidden
#         self.dim_out = dim_out
#         self.n_layers = n_layers
#         self.skip_connections = skip_connections #index of skip connections, starting from 0
#         self.lr = lr
#         self.n_encoders = n_encoders
#         self.n_frequencies = n_frequencies
#         self.sigma = sigma

#         # self.encoder = tcnn.Encoding(n_input_dims=dim_in, encoding_config=config['encoding'])
        
#         # b = rff.functional.sample_b(sigma=10.0, size=self.n_frequencies)
#         # b = rff.functional.sample_b(sigma=10.0, size=self.n_frequencies + (self.dim_in,)).reshape(-1, 4)
#         # self.encoder = rff.layers.GaussianEncoding(b=b)
#         # self.encoder = rff.layers.GaussianEncoding(sigma=10.0, input_size=(self.dim_in), encoded_size=self.n_frequencies)
#         # self.encoder = rff.layers.GaussianEncoding(sigma=self.sigma, input_size=(self.dim_in - 1), encoded_size=self.n_frequencies)
#         # self.encoder_t = rff.layers.GaussianEncoding(sigma=self.sigma_t, input_size=1, encoded_size=self.n_frequencies_t)
#         # self.encoding_dim_out = self.n_frequencies * 2 + self.n_frequencies_t * 2
#         self.encoder_list = torch.nn.ModuleList(rff.layers.GaussianEncoding(sigma=10.0, input_size=(self.dim_in), encoded_size=self.n_frequencies * (i + 1)) for i in range(self.n_encoders))
        
#         self.encoding_dim_out = 0
#         for i in range(self.n_encoders):
#             self.encoding_dim_out += ((i + 1) * self.n_frequencies)
            
#         self.encoding_dim_out *= 2
        
#         self.decoder = torch.nn.ModuleList()
#         for i in range(self.n_layers):
#             if i == 0:
#                 in_features = self.encoding_dim_out
#             elif i in self.skip_connections:
#                 in_features = self.encoding_dim_out + self.dim_hidden
#             else:
#                 in_features = self.dim_hidden
#             block = torch.nn.Sequential(
#                 torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(in_features=in_features, out_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden), n_power_iterations=4, eps=1e-12, dim=None),
#                 # torch.nn.Linear(in_features=in_features, out_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden),
#                 torch.nn.BatchNorm1d(num_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden), #you can do batchnorm 3D + 1D and cat after
#                 # torch.nn.ReLU()
#                 torch.nn.GELU()
#             )
#             self.decoder.append(block)
            

#     def forward(self, x):
#         z = self.encoder_list[0](x)
#         for i in range(1, self.n_encoders):
#             z = torch.hstack((z, self.encoder_list[i](x)))
#         for idx, layer in enumerate(self.decoder):
#             z = layer(z)
#         return z

#     def configure_optimizers(self):
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5) #weight_decay=1e-5
#         return self.optimizer

#     def training_step(self, batch, batch_idx):
#         '''
#         TODO: separate reg in second half of training only ?
#         '''
#         x, y = batch

#         y_pred = self.forward(x)
#         loss = F.mse_loss(y_pred, y)
#         self.log("train_loss", loss)
#         return loss
    
# siren = Siren(1, 1)

# out = siren(x.unsqueeze(-1))
# out = out.detach().numpy()

# plt.plot(range(len(x)), [func(x) for x in x])
# plt.savefig('out.png')
# plt.clf()

# #extract i

# #Fourier

# #Hash enco

# def func(x):
#     return 1 if x > 0.2 and x < 0.6 else 0

# enco1 = GaussianFourierFeatureTransform(2)

# enco2 = rff.layers.GaussianEncoding(sigma=10.0, input_size=2, encoded_size=256)

# for idx, row in enumerate(im):
#     im[idx] = np.sin(idx)

# plt.imshow(im)
# plt.savefig('out.png')



# axes = []
# for s in config.image_shape:
#     axes.append(torch.linspace(0, 1, s))

# mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

# points = mgrid[:,:,::2,:].reshape(-1, 3)
# values = data[..., ::2].reshape(-1, 1)
# xi = mgrid[:,:,1::2,:].reshape(-1, 3)

# interpolation = griddata(points, values, xi, method='linear')