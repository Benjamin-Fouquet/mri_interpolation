'''
Implementation 'Continuous Longitudinal Fetus Brain Atlas Construction via Implicit Neural Representation' using tinycuda

current state: Test if the differential encoding and reordering of T is beneficial for resutlts. Maybe try first 1 encoder but with reordered T, then dual encoder
'''
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
from models import Modulator, ModulatedSirenNet

torch.manual_seed(1337)

@dataclass
class BaseConfig:
    checkpoint_path: Optional[str] = None #'lightning_logs/version_66/checkpoints/epoch=49-step=5000.ckpt'
    log: str = None
    # image_path: str = '/mnt/Data/FetalAtlas/template_T2.nii.gz'
    image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_shape = nib.load(image_path).shape
    batch_size: int = 50000 #~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 40
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None 
    # Network parameters
    encoder_type: str = 'rff' #  
    w0: float = 2.8
    n_frequencies: int = 352 #for classic, n_out = 2 * n_freq. For tcnn, n_out = 2 * n_freq * dim_in
    sigma: float = 10
    w0_t: float = 26.5
    n_frequencies_t: int = 15
    sigma_t: float = 4
    dim_in: int = len(image_shape)
    dim_hidden: int = 256 
    dim_out: int = 1
    num_layers: int = 4
    skip_connections: tuple = () #(5, 11,)
    lr: float = 0.005  # G requires training with a custom lr, usually lr * 0.1 #10-4 for Siren
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
    parser.add_argument("--dim_hidden", help="size of hidden layers", type=int, required=False)
    parser.add_argument("--num_layers", help="number of layers", type=int, required=False)
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
    
class RealGaborLayer(torch.nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
            dim_in: Input features
            dim_out; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, dim_in, dim_out, bias=True,
                 is_first=False, w0=30.0, c=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = w0
        self.scale_0 = c
        self.is_first = is_first
        
        self.dim_in = dim_in
        
        self.freqs = torch.nn.Linear(dim_in, dim_out, bias=bias)
        self.scale = torch.nn.Linear(dim_in, dim_out, bias=bias)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))
    
class ComplexGaborLayer(torch.nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            dim_in: Input features
            dim_out; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, dim_in, dim_out, bias=True,
                 is_first=False, w0=10.0, c=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = w0
        self.scale_0 = c
        self.is_first = is_first
        
        self.dim_in = dim_in
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = torch.nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = torch.nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = torch.nn.Linear(dim_in,
                                dim_out,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
    
class GaborNet(pl.LightningModule):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        n_layers,
        sigma,
        w0,
        lr,
        *args,
        **kwargs
    ):
        super().__init__()
        self.layer_cls = RealGaborLayer
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.lr = lr
        self.sigma = sigma
        self.w0 = w0
        
        layers = []
        for i in range(self.n_layers):
            layers.append(self.layer_cls(dim_in=self.dim_in if i == 0 else self.dim_hidden, dim_out=self.dim_out if i == (n_layers -1) else self.dim_hidden, c=self.sigma, w0=self.w0))
            
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) #weight_decay=1e-5
        return self.optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log("train_loss", loss)
        self.final_loss = float(loss.detach().cpu().numpy()) #parameter used for optuna
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return y_pred

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
        w0,
        n_frequencies,
        sigma,
        w0_t,
        n_frequencies_t,
        sigma_t,
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
        self.w0 = w0
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.w0_t = w0_t
        self.n_frequencies_t = n_frequencies_t
        self.sigma_t = sigma_t
        self.encoder_type = encoder_type
        self.second_training = False


        if self.encoder_type == 'gabor':
            self.encoder = torch.nn.Sequential(RealGaborLayer(dim_in=(self.dim_in - 1), dim_out=self.n_frequencies, is_first=True, w0=self.w0, c=self.sigma), RealGaborLayer(dim_in=self.n_frequencies, dim_out=self.n_frequencies, w0=self.w0, c=self.sigma))
            self.encoder_t = RealGaborLayer(dim_in=1 ,dim_out=self.n_frequencies_t, is_first=True, w0=self.w0_t, c=self.sigma_t)
            self.encoding_dim_out = self.n_frequencies + self.n_frequencies_t
            # self.encoder = torch.nn.Sequential(ComplexGaborLayer(in_features=self.dim_in, out_features=self.n_frequencies, is_first=True) , ComplexGaborLayer(in_features=self.n_frequencies, out_features=self.n_frequencies), ComplexGaborLayer(in_features=self.n_frequencies, out_features=1))
        if self.encoder_type == 'siren':
            self.encoder = torch.nn.Sequential(Siren(dim_in=(self.dim_in - 1),dim_out=self.n_frequencies, is_first=True, w0=self.w0, c=self.sigma), Siren(dim_in=self.n_frequencies ,dim_out=self.n_frequencies, is_first=False, w0=self.w0, c=self.sigma))
            self.encoder_t = Siren(dim_in=1 ,dim_out=self.n_frequencies_t, is_first=True, w0=self.w0_t, c=self.sigma_t)
            self.encoding_dim_out = self.n_frequencies + self.n_frequencies_t
            
        elif self.encoder_type == 'modulated_siren':
            self.encoder = ModulatedSirenNet(dim_in=(self.dim_in - 1), dim_hidden=self.n_frequencies, dim_out=self.n_frequencies, num_layers=2, w0_initial=self.w0, w0=self.w0, c=self.sigma)
            self.encoder_t = Siren(dim_in=1 ,dim_out=self.n_frequencies_t, is_first=True, w0=self.w0_t, c=self.sigma_t)
            self.encoding_dim_out = self.n_frequencies + self.n_frequencies_t
            
        elif self.encoder_type == 'tcnn': #if tcnn is especially required, set it TODO: getattr more elegant
            self.encoder = tcnn.Encoding(n_input_dims=(self.dim_in - 1), encoding_config={'otype': 'Frequency', 'n_frequencies': self.n_frequencies}, dtype=torch.float32)
            self.encoder_t = tcnn.Encoding(n_input_dims=1, encoding_config={'otype': 'Frequency', 'n_frequencies': self.n_frequencies_t}, dtype=torch.float32)
            self.encoding_dim_out = (self.n_frequencies * 2 * (self.dim_in - 1) + self.n_frequencies_t * 2)
        
        elif self.encoder_type == 'rff': #fallback to classic
            # b = rff.functional.sample_b(sigma=10.0, size=self.n_frequencies)
            # b = rff.functional.sample_b(sigma=10.0, size=self.n_frequencies + (self.dim_in,)).reshape(-1, 4)
            # self.encoder = rff.layers.GaussianEncoding(b=b)
            # self.encoder = rff.layers.GaussianEncoding(sigma=10.0, input_size=(self.dim_in), encoded_size=self.n_frequencies)
            self.encoder = rff.layers.GaussianEncoding(sigma=self.sigma, input_size=(self.dim_in - 1), encoded_size=self.n_frequencies)
            self.encoder_t = rff.layers.GaussianEncoding(sigma=self.sigma_t, input_size=1, encoded_size=self.n_frequencies_t)
            self.encoding_dim_out = self.n_frequencies * 2 + self.n_frequencies_t * 2
        else:
            print('encoder type not recognized')
            # self.encoder =  GaussianFourierFeatureTransform(num_input_channels=(self.dim_in) , mapping_size=n_frequencies)
            # self.encoder =  GaussianFourierFeatureTransform(num_input_channels=(self.dim_in - 1) , mapping_size=n_frequencies)
            # self.encoder_t =  GaussianFourierFeatureTransform(num_input_channels=1 ,mapping_size=n_frequencies_t)
            
        ##Legacy enco out, reimplement if needed     
        # self.encoding_dim_out = (self.n_frequencies * 2 * (self.dim_in)) if isinstance(self.encoder, tcnn.Encoding) else (self.n_frequencies * 2)  #version for non separated dimensions
        
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
        coords = x[:, :(self.dim_in - 1)]
        t = x[:, -1].unsqueeze(-1)
        x = torch.hstack((self.encoder(coords), self.encoder_t(t)))
        skip = x.clone()
        for idx, layer in enumerate(self.decoder):
            if idx in self.skip_connections:
                x = torch.hstack((skip, x))
            x = layer(x)
        return x
        
        # x = self.encoder(x)
        # for idx, layer in enumerate(self.decoder):
        #     x = layer(x)
        # return x

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
    
    def lr_schedulers(self) -> LRSchedulerTypeUnion | List[LRSchedulerTypeUnion] | None:
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=10, verbose=True)
        return self.scheduler
    
    def on_train_end(self) -> None:
        writer = SummaryWriter(log_dir=self.logger.log_dir)
        writer.add_text(text_string=str(config), tag='configuration')
        writer.close()
        # print(str(model.lr_schedulers().get_last_lr()))
        
mri_image = nib.load(config.image_path)

data = mri_image.get_fdata(dtype=np.float32)
data = data[:,:,3,:] #optional line for doing 3D and accelerate prototyping
config.image_shape = data.shape
config.dim_in = len(data.shape)

#interpolation tests
# data = data[..., ::2]

if config.checkpoint_path is not None:
    model = FreqMLP.load_from_checkpoint(
                    config.checkpoint_path,
                    dim_in=config.dim_in, 
                    dim_hidden=config.dim_hidden, 
                    dim_out=config.dim_out, 
                    n_layers=config.num_layers, 
                    skip_connections=config.skip_connections,
                    w0=config.w0,
                    n_frequencies=config.n_frequencies,
                    sigma=config.sigma,
                    w0_t=config.w0_t,
                    n_frequencies_t=config.n_frequencies_t,
                    sigma_t=config.sigma_t,                
                    encoder_type=config.encoder_type,
                    lr=config.lr)  
    
else:
  
    model = FreqMLP(dim_in=config.dim_in, 
                    dim_hidden=config.dim_hidden, 
                    dim_out=config.dim_out, 
                    n_layers=config.num_layers, 
                    skip_connections=config.skip_connections,
                    w0=config.w0,
                    n_frequencies=config.n_frequencies,
                    sigma=config.sigma,
                    w0_t=config.w0_t,
                    n_frequencies_t=config.n_frequencies_t,
                    sigma_t=config.sigma_t,                
                    encoder_type=config.encoder_type,
                    lr=config.lr)  
 

Y = torch.FloatTensor(data).reshape(-1, 1)
Y = Y / Y.max()

axes = []
for s in config.image_shape:
    axes.append(torch.linspace(0, 1, s))

mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

coords = torch.FloatTensor(mgrid)

# if data.shape[-1] < 15:
#     if config.dim_in ==3:
#         coords = coords[:,:,::2,:]
#     if config.dim_in ==4:
#         coords = coords[:,:,:,::2,:]

X = coords.reshape(len(Y), config.dim_in)

dataset = torch.utils.data.TensorDataset(X, Y)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
   
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
            
im = pred.reshape(data.shape)
im = im.detach().cpu().numpy()
im = np.array(im, dtype=np.float32)
if len(im.shape) == 2:
    plt.imshow(im.T)
    plt.savefig(filepath + 'pred.png')
else:
    nib.save(nib.Nifti1Image(im, affine=np.eye(4)), filepath + 'pred.nii.gz')


interp_shapes = [(352, 352, 30), (352, 352, 60)]
# interp_shapes = [(117, 159, 126, 30)]
#ugly loop as placeholder
for shape in interp_shapes:    
    #dense grid
    config.interp_shape = shape
    Y_interp = torch.zeros((np.prod(config.interp_shape), 1))

    axes = []
    for s in config.interp_shape:
        axes.append(torch.linspace(0, 1, s))
            
    mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

    coords = torch.FloatTensor(mgrid)
    X_interp = coords.reshape(len(Y_interp), config.dim_in)    

    interp_dataset = torch.utils.data.TensorDataset(X_interp, Y_interp)
    interp_loader = torch.utils.data.DataLoader(interp_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    #create an interpolation
    interp = torch.concat(trainer.predict(model, interp_loader))

    interp_im = interp.reshape(config.interp_shape)
        
    interp_im = interp_im.detach().cpu().numpy()
    interp_im = np.array(interp_im, dtype=np.float32)
    nib.save(nib.Nifti1Image(interp_im, affine=np.eye(4)), filepath + f'interpolation{shape}.nii.gz')

config.export_to_txt(file_path=filepath)
            



