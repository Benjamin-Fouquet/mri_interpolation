from typing import List, Optional, Union
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
import tinycudann as tcnn 
import torch
import pytorch_lightning as pl 
import torch.nn.functional as F
import json
import nibabel as nib 
from dataclasses import dataclass
import os
from types import MappingProxyType
import numpy as np
import math
import rff
import argparse
from torch.utils.tensorboard import SummaryWriter
from math import pi
import torch.utils.data
import matplotlib.pyplot as plt

torch.manual_seed(1337)

@dataclass
class BaseConfig:
    checkpoint_path = None #'lightning_logs/version_25/checkpoints/epoch=49-step=11200.ckpt'
    log: str = None
    # image_path: str = '/mnt/Data/FetalAtlas/template_T2.nii.gz'
    image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_shape = nib.load(image_path).shape
    batch_size: int = 30000 #~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 50
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None 
    # Network parameters
    n_encoders: int = 9 
    n_frequencies: int = 4  #for classic, n_out = 2 * n_freq. For tcnn, n_out = 2 * n_freq * dim_in
    sigma: float = 2.0
    dim_in: int = len(image_shape)
    dim_hidden: int = 128
    dim_out: int = 1
    num_layers: int = 4
    skip_connections: tuple = () #(5, 11,)
    lr: float = 1e-3  # G requires training with a custom lr, usually lr * 0.1 
    interp_factor: int = 2

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch size", type=int, required=False)
    parser.add_argument("--epochs", help="Number of epochs", type=int, required=False)
    parser.add_argument("--image_path", help="path of image", type=str, required=False)
    parser.add_argument("--encoder_type", help="tcnn or classic", type=str, required=False)
    parser.add_argument("--n_frequencies", help="number of encoding frequencies", type=int, required=False)
    parser.add_argument("--n_frequencies_t", help="number of encoding frequencies for time", type=int, required=False)
    args = parser.parse_args()

def export_to_txt(dict: dict, file_path: str = "") -> None:
    '''
    Helper function to export dictionary to text file
    '''
    with open(file_path + "config.txt", "a+") as f:
        for key in dict:
            f.write(str(key) + " : " + str(dict[key]) + "\n")

config = BaseConfig()

# parsed argument -> config
for key in args.__dict__:
    if args.__dict__[key] is not None:
        config.__dict__[key] = args.__dict__[key]


#freq encoding tiny cuda
class FreqMLP(pl.LightningModule):
    '''
    Lightning module for HashMLP. 
    '''
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        n_layers,
        skip_connections,
        n_encoders,
        n_frequencies,
        sigma,
        lr,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.skip_connections = skip_connections #index of skip connections, starting from 0
        self.lr = lr
        self.n_encoders = n_encoders
        self.n_frequencies = n_frequencies
        self.sigma = sigma

        # self.encoder = tcnn.Encoding(n_input_dims=dim_in, encoding_config=config['encoding'])
        
        # b = rff.functional.sample_b(sigma=10.0, size=self.n_frequencies)
        # b = rff.functional.sample_b(sigma=10.0, size=self.n_frequencies + (self.dim_in,)).reshape(-1, 4)
        # self.encoder = rff.layers.GaussianEncoding(b=b)
        # self.encoder = rff.layers.GaussianEncoding(sigma=10.0, input_size=(self.dim_in), encoded_size=self.n_frequencies)
        # self.encoder = rff.layers.GaussianEncoding(sigma=self.sigma, input_size=(self.dim_in - 1), encoded_size=self.n_frequencies)
        # self.encoder_t = rff.layers.GaussianEncoding(sigma=self.sigma_t, input_size=1, encoded_size=self.n_frequencies_t)
        # self.encoding_dim_out = self.n_frequencies * 2 + self.n_frequencies_t * 2
        self.encoder_list = torch.nn.ModuleList(rff.layers.GaussianEncoding(sigma=10.0, input_size=(self.dim_in), encoded_size=self.n_frequencies * (i + 1)) for i in range(self.n_encoders))
        
        self.encoding_dim_out = 0
        for i in range(self.n_encoders):
            self.encoding_dim_out += ((i + 1) * self.n_frequencies)
            
        self.encoding_dim_out *= 2
        
        self.decoder = torch.nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                in_features = self.encoding_dim_out
            elif i in self.skip_connections:
                in_features = self.encoding_dim_out + self.dim_hidden
            else:
                in_features = self.dim_hidden
            block = torch.nn.Sequential(
                torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(in_features=in_features, out_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden), n_power_iterations=4, eps=1e-12, dim=None),
                # torch.nn.Linear(in_features=in_features, out_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden),
                torch.nn.BatchNorm1d(num_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden), #you can do batchnorm 3D + 1D and cat after
                # torch.nn.ReLU()
                torch.nn.GELU()
            )
            self.decoder.append(block)
            

    def forward(self, x):
        z = self.encoder_list[0](x)
        for i in range(1, self.n_encoders):
            z = torch.hstack((z, self.encoder_list[i](x)))
        for idx, layer in enumerate(self.decoder):
            z = layer(z)
        return z

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5) #weight_decay=1e-5
        return self.optimizer

    def training_step(self, batch, batch_idx):
        '''
        TODO: separate reg in second half of training only ?
        '''
        x, y = batch

        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log("train_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return y_pred
    
        
mri_image = nib.load(config.image_path)

data = mri_image.get_fdata(dtype=np.float32)
data = data[:,:,3,:] #optional line for doing 3D and accelerate prototyping
config.image_shape = data.shape
config.dim_in = len(data.shape)

if config.checkpoint_path is not None:
    model = FreqMLP.load_from_checkpoint(
                    config.checkpoint_path,
                    dim_in=config.dim_in, 
                    dim_hidden=config.dim_hidden, 
                    dim_out=config.dim_out, 
                    n_layers=config.num_layers, 
                    skip_connections=config.skip_connections,
                    n_encoders=config.n_encoders,
                    n_frequencies=config.n_frequencies,
                    sigma=config.sigma,
                    lr=config.lr)  
    
else:
    model = FreqMLP(
                    dim_in=config.dim_in, 
                    dim_hidden=config.dim_hidden, 
                    dim_out=config.dim_out, 
                    n_layers=config.num_layers, 
                    skip_connections=config.skip_connections,
                    n_encoders=config.n_encoders,
                    n_frequencies=config.n_frequencies,
                    sigma=config.sigma,
                    lr=config.lr)  
 

Y = torch.FloatTensor(data).reshape(-1, 1)
Y = Y / Y.max()

axes = []
for s in config.image_shape:
    axes.append(torch.linspace(0, 1, s))

mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

coords = torch.FloatTensor(mgrid)
X = coords.reshape(len(Y), config.dim_in)

dataset = torch.utils.data.TensorDataset(X, Y)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

# #dense grid
Y_interp = torch.zeros((np.prod(config.image_shape) * config.interp_factor, 1))
axes = []
for idx, s in enumerate(config.image_shape):
    if idx == (len(config.image_shape) - 1):
        axes.append(torch.linspace(0, 1, s * config.interp_factor))        
    else:
        axes.append(torch.linspace(0, 1, s))
        
mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

coords = torch.FloatTensor(mgrid)
X_interp = coords.reshape(len(Y_interp), config.dim_in)    

interp_dataset = torch.utils.data.TensorDataset(X_interp, Y_interp)
interp_loader = torch.utils.data.DataLoader(interp_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
   
trainer = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
    precision=32,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)

trainer.fit(model, train_loader)

filepath = model.logger.log_dir + '/'
config.log = str(model.logger.version)

#create a prediction
pred = torch.concat(trainer.predict(model, test_loader))
            
im = pred.reshape(config.image_shape)
im = im.detach().cpu().numpy()
im = np.array(im, dtype=np.float32)
if len(im.shape) == 2:
    plt.imshow(im.T)
    plt.savefig(filepath + 'pred.png')
else:
    nib.save(nib.Nifti1Image(im, affine=np.eye(4)), filepath + 'pred.nii.gz')

# #create an interpolation
interp = torch.concat(trainer.predict(model, interp_loader))
       
if len(data.shape) == 3:     
    interp_im = interp.reshape((mri_image.shape[0], mri_image.shape[1], mri_image.shape[3] * config.interp_factor))
if len(data.shape) == 4:
    interp_im = interp.reshape((mri_image.shape[0], mri_image.shape[1], mri_image.shape[2], mri_image.shape[3] * config.interp_factor))
interp_im = interp_im.detach().cpu().numpy()
interp_im = np.array(interp_im, dtype=np.float32)
nib.save(nib.Nifti1Image(interp_im, affine=np.eye(4)), filepath + 'interpolation.nii.gz')

config.export_to_txt(file_path=filepath)
            



