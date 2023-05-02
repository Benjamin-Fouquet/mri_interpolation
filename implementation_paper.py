'''
Implementation 'Continuous Longitudinal Fetus Brain Atlas Construction via Implicit Neural Representation' using tinycuda
'''
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

torch.manual_seed(1337)

@dataclass
class BaseConfig:
    checkpoint_path = None #'lightning_logs/version_384/checkpoints/epoch=99-step=100.ckpt'
    # image_path: str = '/mnt/Data/FetalAtlas/template_T2.nii.gz'
    image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_shape = nib.load(image_path).shape
    batch_size: int = 100000 #~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 50
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None 
    # Network parameters
    encoder_type: str = None #'tcnn'
    n_frequencies: int = 64  #for classic, n_out = 2 * n_freq. For tcnn, n_out = 2 * n_freq * dim_in
    dim_in: int = len(image_shape)
    dim_hidden: int = 256 #should match n_frequencies
    dim_out: int = 1
    num_layers: int = 18
    skip_connections: tuple = (5, 11,)
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

#utils for siren
def exists(val):
    return val is not None


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Sine(torch.nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(torch.nn.Module):
    '''
    Siren layer
    '''
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

        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias) if use_bias else None
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
        encoder_type,
        n_frequencies,
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
        self.n_frequencies = n_frequencies
        self.encoder_type = encoder_type

        # self.encoder = tcnn.Encoding(n_input_dims=dim_in, encoding_config=config['encoding'])
        if self.encoder_type == 'tcnn': #if tcnn is especially required, set it TODO: getattr more elegant
            #create the dictionary
            self.encoder = tcnn.Encoding(n_input_dims=self.dim_in, encoding_config={'otype': 'Frequency', 'n_frequencies': self.n_frequencies})
        else: #fallback to classic
            self.encoder = rff.layers.GaussianEncoding(sigma=10.0, input_size=self.dim_in, encoded_size=self.n_frequencies)
            
        self.encoding_dim_out = self.n_frequencies * 2 * self.dim_in if isinstance(self.encoder, tcnn.Encoding) else self.n_frequencies * 2
        # self.encoder = torch.nn.Sequential(Siren(dim_in=self.dim_in, dim_out=self.dim_in * 2 * config['encoding']['n_frequencies']), Siren(dim_in=self.dim_in * 2 * config['encoding']['n_frequencies'], dim_out=self.dim_in * 2 * config['encoding']['n_frequencies']))
        # self.decoder = tcnn.Network(n_input_dims=self.encoder.n_output_dims, n_output_dims=dim_out, network_config=config['network'])
        self.decoder = torch.nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                in_features = self.encoding_dim_out
            elif i in self.skip_connections:
                in_features = self.encoding_dim_out + self.dim_hidden
            else:
                in_features = self.dim_hidden
            block = torch.nn.Sequential(
                torch.nn.Linear(in_features=in_features, out_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden),
                torch.nn.BatchNorm1d(num_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden), #you can do batochnorm 3D + 1D and cat after
                torch.nn.ReLU()
            )
            self.decoder.append(block)
            

    def forward(self, x):
        x = self.encoder(x)
        skip = x.clone()
        for idx, layer in enumerate(self.decoder):
            if idx in self.skip_connections:
                x = torch.hstack((skip, x))
            x = layer(x)
        return x 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5) #weight_decay=1e-5
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)

        loss = F.mse_loss(y_pred, y)

        self.log("train_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return y_pred

    
model = FreqMLP(dim_in=config.dim_in, 
                dim_hidden=config.dim_hidden, 
                dim_out=config.dim_out, 
                n_layers=config.num_layers, 
                skip_connections=config.skip_connections,
                n_frequencies=config.n_frequencies,
                encoder_type=config.encoder_type,
                lr=config.lr)

#include batch norm, make a mock dataset (why mock, you can do a normal one)
mri_image = nib.load(config.image_path)

Y = torch.FloatTensor(mri_image.get_fdata(dtype=np.float32)).flatten().unsqueeze(-1)
Y = Y / Y.max()

axes = []
for s in mri_image.shape:
    axes.append(torch.linspace(0, 1, s))

mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

coords = torch.FloatTensor(mgrid)
X = coords.reshape(len(Y), config.dim_in)

dataset = torch.utils.data.TensorDataset(X, Y)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)  
                
trainer = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
    precision=16,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)
# trainer = pl.Trainer(gpus=config.device, max_epochs=config.epochs)
trainer.fit(model, train_loader)

filepath = model.logger.log_dir + '/'

#create a prediction
pred = torch.concat(trainer.predict(model, test_loader))
            
im = pred.reshape(mri_image.shape)
im = im.detach().cpu().numpy()
im = np.array(im, dtype=np.float32)
nib.save(nib.Nifti1Image(im, affine=np.eye(4)), filepath + 'pred.nii.gz')

#t interp * 10
Y_interp = torch.zeros((np.prod(mri_image.shape) * config.interp_factor, 1))

axes = []
for idx, s in enumerate(mri_image.shape):
    if idx == len(mri_image.shape) - 1:
        axes.append(torch.linspace(0, 1, s * config.interp_factor))        
    else:
        axes.append(torch.linspace(0, 1, s))

mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

coords = torch.FloatTensor(mgrid)
X_interp = coords.reshape(len(Y_interp), config.dim_in)    

interp_dataset = torch.utils.data.TensorDataset(X_interp, Y_interp)
interp_loader = torch.utils.data.DataLoader(interp_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

#create an interpolation
interp = torch.concat(trainer.predict(model, interp_loader))
            
interp_im = interp.reshape((mri_image.shape[0], mri_image.shape[1], mri_image.shape[2], mri_image.shape[3] * config.interp_factor))
interp_im = interp_im.detach().cpu().numpy()
interp_im = np.array(interp_im, dtype=np.float32)
nib.save(nib.Nifti1Image(interp_im, affine=np.eye(4)), filepath + 'interpolation.nii.gz')

config.export_to_txt(file_path=filepath)

# conf = {'otype': 'Frequency', 'n_frequencies': 64}

# enc = tcnn.Encoding(n_input_dims=3, encoding_config=conf)
# enc2 = rff.layers.GaussianEncoding(sigma=10.0, input_size=4, encoded_size=64)
# enc(X[:config.batch_size]).shape
