#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import argparse
# import tinycudann as tcnn
import os

class HashMLP(pl.LightningModule):
  def __init__(self, config, dim_in=3, dim_out=1):
    super().__init__()
    self.dim_in = dim_in
    self.dim_out = dim_out

    self.encoding = tcnn.Encoding(n_input_dims=dim_in, encoding_config=config['encoding'])
    self.mlp= tcnn.Network(n_input_dims=self.encoding.n_output_dims, n_output_dims=dim_out, network_config=config['network'])
    self.model = torch.nn.Sequential(self.encoding, self.mlp)

  def forward(self, x):
    return self.model(x)

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


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo TCNN')
  parser.add_argument('-i', '--input', help='Multiple input images (nifti)', action='append', type=str, required=True)
  parser.add_argument('-o', '--output', help='Output image (nifti)', type=str, required=True)
  parser.add_argument('-m', '--model', help='Pytorch lightning (ckpt file) trained model', type=str, required=False)
  parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 4096)    
  parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required=False, default = 10)  
  parser.add_argument('-n', '--neurons', help='Number of neurons in MLP layers', type=int, required=False, default = 128)  
  parser.add_argument('-l', '--layers', help='Number of layers in MLP', type=int, required=False, default = 2)  
  parser.add_argument('-f', '--features', help='Number of features per level (hash grid)', type=int, required=False, default = 2)  
  parser.add_argument(      '--levels', help='Number of levels (hash grid)', type=int, required=False, default = 8)  
  parser.add_argument(      '--log2_hashmap_size', help='Log2 hashmap size (hash grid)', type=int, required=False, default = 15) #15:nvidia, 19: nesvor  
  parser.add_argument(      '--base', help='Base resolution', type=int, required=False, default = 16)    
  parser.add_argument(      '--encodings', help='Output encoding image (nifti)', type=str, required=False)

  args = parser.parse_args()

  image_list = ['data/HCP/100307_T1.nii.gz', 'data/HCP/100307_T2.nii.gz']
  n_images = len(image_list)
  output_file = args.output
  model_file = args.model

  num_epochs = 20
  batch_size = 4000
  num_workers = os.cpu_count()

  #Read first image
  print('Reading : '+image_list[0])
  image = nib.load(image_list[0])
  data = image.get_fdata()

  #Create grid
  dim = 3
  nx = data.shape[0]
  ny = data.shape[1]
  nz = data.shape[2]
  nmax = np.max([nx,ny,nz])

  x = torch.linspace(0, 1, steps=nx)
  y = torch.linspace(0, 1, steps=ny)
  z = torch.linspace(0, 1, steps=nz)
  
  mgrid = torch.stack(torch.meshgrid(x,y,z,indexing='ij'), dim=-1)
  
  #Convert to X=(x,y,z) and Y=intensity
  X = torch.Tensor(mgrid.reshape(-1,dim))
  Y = torch.Tensor(data.flatten())
  
  #Normalize intensities between [-1,1]
  Y = (Y - torch.min(Y)) / (torch.max(Y) - torch.min(Y)) * 2 - 1
  Y = torch.reshape(Y, (-1,1))
  
  #Add other images
  for i in range(1,n_images):
    print('Reading : '+image_list[i])
    image = nib.load(image_list[i])
    data = image.get_fdata()
    Ytmp = torch.Tensor(data.flatten())
    Ytmp = (Ytmp - torch.min(Ytmp)) / (torch.max(Ytmp) - torch.min(Ytmp)) * 2 - 1
    Ytmp = torch.reshape(Ytmp, (-1,1))
    Y = torch.concat((Y,Ytmp),dim=1)

  print(Y.shape)    

  #Pytorch dataloader
  dataset = torch.utils.data.TensorDataset(X,Y)
  loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

  #Training
  base_resolution = args.base
  n_levels = args.levels
  b = np.exp((np.log(nmax)-np.log(base_resolution))/(n_levels-1))

  #https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
  config = {
  "encoding": {
		"otype": "HashGrid",
		"n_levels": n_levels,
		"n_features_per_level": args.features,
		"log2_hashmap_size": args.log2_hashmap_size,
		"base_resolution": base_resolution,
		"per_level_scale": b#1.3819#1.5
	},
	"network": {
		"otype": "FullyFusedMLP",
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": args.neurons,
		"n_hidden_layers": args.layers
	}
  }

  net = HashMLP(config = config, dim_in=3, dim_out=n_images)
  trainer = pl.Trainer(max_epochs=num_epochs, precision=16)

  #net = torch.compile(net)   #Not working for old GPU like Titan
  trainer.fit(net, loader)

  if args.model is not None:
    trainer.save_checkpoint(model_file) 
    
  test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True) #remove shuffling
  yhat = torch.concat(trainer.predict(net, test_loader))

  print(yhat.shape)
  output = np.float32(yhat.cpu().detach().numpy().reshape((nx,ny,nz,n_images)))
  nib.save(nib.Nifti1Image(output, image.affine), output_file)     

  if args.encodings is not None:
    X = X.to(device='cuda')
    net = net.to(device='cuda')
    enc = net.encoding(X)

    n_features = args.levels * args.features

    data4d = enc.cpu().detach().numpy().reshape((nx,ny,nz,n_features))
    nib.save(nib.Nifti1Image(np.float32(data4d),image.affine), args.encodings)