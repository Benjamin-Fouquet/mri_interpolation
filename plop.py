
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:30:59 2022
@author: rousseau
"""
from os.path import expanduser
home = expanduser("~")
import nibabel as nib
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os

#%% Code from SIREN repo modified for lightning
import math
from einops import rearrange

def exists(val):
  return val is not None

def cast_tuple(val, repeat = 1):
  return val if isinstance(val, tuple) else ((val,) * repeat)
  
class Sine(nn.Module):
  def __init__(self, w0 = 1.):
    super().__init__()
    self.w0 = w0
  def forward(self, x):
    return torch.sin(self.w0 * x)

# siren layer
class Siren(nn.Module):
  def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
    super().__init__()
    self.dim_in = dim_in
    self.is_first = is_first

    weight = torch.zeros(dim_out, dim_in)
    bias = torch.zeros(dim_out) if use_bias else None
    self.init_(weight, bias, c = c, w0 = w0)

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
    out =  F.linear(x, self.weight, self.bias)
    out = self.activation(out)
    return out
  
# siren network
class SirenNet(pl.LightningModule):
  def __init__(self, dim_in=3, dim_hidden=128, dim_out=1, num_layers=2, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
    super().__init__()
    self.num_layers = num_layers
    self.dim_hidden = dim_hidden

    self.layers = nn.ModuleList([])
    for ind in range(num_layers):
        is_first = ind == 0
        layer_w0 = w0_initial if is_first else w0
        layer_dim_in = dim_in if is_first else dim_hidden

        self.layers.append(Siren(
            dim_in = layer_dim_in,
            dim_out = dim_hidden,
            w0 = layer_w0,
            use_bias = use_bias,
            is_first = is_first
        ))

    final_activation = nn.Identity() if not exists(final_activation) else final_activation
    self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

  def forward(self, x, mods = None):
    mods = cast_tuple(mods, self.num_layers)

    for layer, mod in zip(self.layers, mods):
      x = layer(x)

      if exists(mod):
        x *= rearrange(mod, 'd -> () d')

    return self.last_layer(x)

  def training_step(self, batch, batch_idx):
    x,y = batch    
    z = self(x)

    loss = F.mse_loss(z, y)

    self.log('train_loss', loss)
    return loss

  def predict_step(self, batch, batch_idx):
    x,y = batch    
    return self(x)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    return optimizer
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo SIREN')
  parser.add_argument('-i', '--input', help='Input image (nifti)', type=str, required=True)
  parser.add_argument('-o', '--output', help='Output image (nifti)', type=str, required=True)
  parser.add_argument('-m', '--model', help='Pytorch lightning (ckpt file) trained model', type=str, required=False)
  parser.add_argument('-n', '--neurons', help='Number of hidden neurons', type=int, required=False, default = 512)
  parser.add_argument('-l', '--layers', help='Number of layers', type=int, required=False, default = 5)  
  parser.add_argument('-w', '--w0', help='Value of w_0', type=float, required=False, default = 30)  
  parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required=False, default = 10)  
  parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 1024)    
  parser.add_argument('--workers', help='Number of workers (multiprocessing). By default: the number of CPU', type=int, required=False, default = -1)

  args = parser.parse_args()

  dim_hidden = args.neurons
  num_layers = args.layers
  w0 = args.w0
  num_epochs = args.epochs
  model_file = args.model
  batch_size = args.batch_size
  image_file = args.input
  output_file = args.output
  num_workers = args.workers
  if num_workers == -1:
    num_workers = os.cpu_count()
  
  #Read image
  image = nib.load(image_file)
  data = image.get_fdata()

  #Create grid
  dim = 3
  x = torch.linspace(-1, 1, steps=data.shape[0])
  y = torch.linspace(-1, 1, steps=data.shape[1])
  z = torch.linspace(-1, 1, steps=data.shape[2])
  
  mgrid = torch.stack(torch.meshgrid(x,y,z), dim=-1)
  
  #Convert to X=(x,y,z) and Y=intensity
  X = torch.Tensor(mgrid.reshape(-1,dim))
  Y = torch.Tensor(data.flatten())
  
  #Normalize intensities between [-1,1]
  Y = (Y - torch.min(Y)) / (torch.max(Y) - torch.min(Y)) * 2 - 1
  Y = torch.reshape(Y, (-1,1))
  
  #Pytorch dataloader
  dataset = torch.utils.data.TensorDataset(X,Y)
  loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

  #Training
  net = SirenNet(dim_in=3, dim_hidden=dim_hidden, dim_out=1, num_layers=num_layers, w0 = w0)
  trainer = pl.Trainer(gpus=1, max_epochs=num_epochs)
  trainer.fit(net, loader)

  if args.model is not None:
    trainer.save_checkpoint(model_file) 
    
  #%% Load trained model (just to check that loading is working) and do the prediction using lightning trainer (for batchsize management)
  #net = SirenNet().load_from_checkpoint(model_file, dim_in=3, dim_hidden=dim_hidden, dim_out=1, num_layers=num_layers, w0 = w0)

  batch_size_test = batch_size * 2 
  test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_test, num_workers=num_workers) #remove shuffling
  yhat = torch.concat(trainer.predict(net, test_loader))

  output = yhat.cpu().detach().numpy().reshape(data.shape)
  nib.save(nib.Nifti1Image(output, image.affine), output_file)   