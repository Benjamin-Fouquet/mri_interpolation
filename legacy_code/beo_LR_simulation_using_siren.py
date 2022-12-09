# -*- coding: utf-8 -*-

import argparse
import numpy as np
import nibabel
from scipy.ndimage import map_coordinates

import torch
import pytorch_lightning as pl

# Objective: simulation LR image using a given PSF
# Inputs: a reference (HR) image and a LR image (created using ITK-based resampling)
# We use ITK resampling because it's a simple way to obtain the new pixel coordinates of LR image 
# Otherwise, we have to compute new coordinates depending on image resolutions (HR and LR)

##MODIFIED SIREN ON PREDICT? DO NOT REUSE
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
from einops import rearrange

#Siren and utils#

def exists(val):
    return val is not None


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# siren layer
class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out

# siren network
class SirenNet(pl.LightningModule):
    def __init__(
        self,
        dim_in=3,
        dim_hidden=64,
        dim_out=1,
        num_layers=4,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
        lr=1e-4
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.losses = []
        self.lr = lr

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(
                Siren(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                )
            )

        final_activation = (
            nn.Identity() if not exists(final_activation) else final_activation
        )
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
        )

    def forward(self, x, mods=None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, "d -> () d")

        return self.last_layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)

        loss = F.mse_loss(z, y)
        self.losses.append(loss.detach().cpu().numpy())

        self.log("train_loss", loss)
        return loss

    def predict_step(self, x, batch_idx):

        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def set_parameters(self, theta):
        '''
        Manually set parameters using matching theta, not foolproof
        '''
        p_dict = self.state_dict()
        for p, thet in zip(p_dict, theta):
            p_dict[p] = thet.data
        self.load_state_dict(p_dict)
        self.eval()
        self.train() #supposed to be important when you set parameters or load state


# siren network
class FourrierNet(pl.LightningModule):
    def __init__(
        self,
        dim_in=3,
        dim_hidden=64,
        dim_out=1,
        num_layers=4,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.losses = []

        self.layers = nn.ModuleList([])
        ###
        for ind in range(num_layers):
            if ind == 0:
                self.layers.append(
                Siren(
                    dim_in=dim_in,
                    dim_out=dim_hidden,
                    w0=w0_initial,
                    use_bias=use_bias,
                    is_first=ind,
                )
            )
            else:
                self.layers.append(nn.Linear(in_features=dim_hidden, out_features=dim_hidden))

        final_activation = (
            nn.Identity() if not exists(final_activation) else final_activation
        )
        self.last_layer = nn.Linear(
            in_features=dim_hidden,
            out_features=dim_out,
        )

    def forward(self, x, mods=None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, "d -> () d")

        return self.last_layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)

        loss = F.mse_loss(z, y)
        self.losses.append(loss.detach().cpu().numpy())

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def set_parameters(self, theta):
        '''
        Manually set parameters using matching theta, not foolproof
        '''
        p_dict = self.state_dict()
        for p, thet in zip(p_dict, theta):
            p_dict[p] = thet.data
        self.load_state_dict(p_dict)
        self.eval()
        self.train() #supposed to be important when you set parameters or load state

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('-r', '--ref',   help='Reference Image filename (i.e. ground truth) (required)', type=str, required = True)
  parser.add_argument('-i', '--input', help='Low-resolution image filename (required), created using ITK-based Resampling', type=str, required = True)
  parser.add_argument('-m', '--model', help='Pytorch lightning (ckpt file) trained model', type=str, required=True)
  parser.add_argument('-o', '--output', help='Low-resolution simulated image filename (required)', type=str, required = True)

  args = parser.parse_args()

  #Load images
  HRimage = nibabel.load(args.ref)
  LRimage = nibabel.load(args.input)

  #Get image resolution
  HRSpacing = np.float32(np.array(HRimage.header['pixdim'][1:4]))  
  LRSpacing = np.float32(np.array(LRimage.header['pixdim'][1:4]))  

  #Pre-compute PSF values
  #PSF is a box centered around an observed pixel of LR image
  #The size of the box is set as the size of a LR pixel (expressed in voxel space)
  n_samples = 5
  psf_sx = np.linspace(-0.5,0.5,n_samples)
  psf_sy = np.linspace(-0.5,0.5,n_samples)
  psf_sz = np.linspace(-0.5,0.5,n_samples)

  #Define a set of points for PSF values using meshgrid
  #https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
  psf_x, psf_y, psf_z = np.meshgrid(psf_sx, psf_sy, psf_sz, indexing='ij')

  #Define gaussian kernel as PSF model
  sigma = 1.0 / 2.3548 #could be anisotropic to reflect MRI sequences (see Kainz et al.)
  def gaussian(x,sigma):
    return np.exp(-x*x/(2*sigma*sigma))  

  psf = gaussian(psf_x,sigma) * gaussian(psf_y, sigma) * gaussian(psf_z,sigma) 
  psf = psf / np.sum(psf)

  #Get data
  HRdata = HRimage.get_fdata()
  LRdata = LRimage.get_fdata()
  outputdata = np.zeros(LRdata.shape)

  #Define transforms
  #This is where we could add slice-by-slice transform
  LR_to_world = LRimage.affine
  world_to_HR = np.linalg.inv(HRimage.affine)
  LR_to_HR = world_to_HR @ LR_to_world

  #PSF coordinates in LR image
  psf_coordinates_in_LR = np.ones((4,psf.size))

  #SIREN stuff
  model_file = args.model
  net = SirenNet().load_from_checkpoint(model_file, dim_in=3, dim_hidden=512, dim_out=1, num_layers=5, w0 = 30)
  trainer = pl.Trainer(gpus=1, max_epochs=1, precision=32)
  psf_coordinates_in_HR_siren = np.ones((3,psf.size))

  #Loop over LR pixels (i,j,k)
  for i in range(LRdata.shape[0]):
    for j in range(LRdata.shape[1]):
      for k in range(LRdata.shape[2]):

        #coordinates of PSF box around current pixel
        psf_coordinates_in_LR[0,:] = psf_x.flatten() + i
        psf_coordinates_in_LR[1,:] = psf_y.flatten() + j
        psf_coordinates_in_LR[2,:] = psf_z.flatten() + k

        #Transform PSF grid to HR space
        psf_coordinates_in_HR = LR_to_HR @ psf_coordinates_in_LR

        #coordinate normalization into [-1,1] to be compatible with SIREN
        psf_coordinates_in_HR_siren[0,:] = (psf_coordinates_in_HR[0,:] / HRdata.shape[0] -0.5)*2 
        psf_coordinates_in_HR_siren[1,:] = (psf_coordinates_in_HR[1,:] / HRdata.shape[1] -0.5)*2 
        psf_coordinates_in_HR_siren[2,:] = (psf_coordinates_in_HR[2,:] / HRdata.shape[2] -0.5)*2 

        #Get interpolated values at psf points in HR
        #interp_values = map_coordinates(HRdata,psf_coordinates_in_HR[0:3,:],order=0,mode='constant',cval=np.nan,prefilter=False)
        x = torch.Tensor(psf_coordinates_in_HR_siren[0:3,:].T)

        interp_values = torch.Tensor(trainer.predict(net, x)).cpu().detach().numpy()
        
        #Compute new weigthed value of LR pixel
        outputdata[i,j,k] = np.sum(psf.flatten()*interp_values)

  
  nibabel.save(nibabel.Nifti1Image(outputdata, LRimage.affine),args.output)    