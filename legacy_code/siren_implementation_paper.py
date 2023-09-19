'''
Implementation 'Continuous Longitudinal Fetus Brain Atlas Construction via Implicit Neural Representation' , test version with gabor wavelets

current state: Test if the differnetial encoding and reordering of T is beneficial for resutlts. Maybe try first 1 encoder but with reordered T, then dual encoder
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
import torch.nn
from functools import lru_cache
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

torch.manual_seed(1337)

@dataclass
class BaseConfig:
    checkpoint_path: Optional[str] = None #'lightning_logs/version_269/checkpoints/epoch=285-step=16874.ckpt' #'lightning_logs/version_384/checkpoints/epoch=99-step=100.ckpt'
    # image_path: str = '/mnt/Data/FetalAtlas/template_T2.nii.gz'
    image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_shape = nib.load(image_path).shape
    batch_size: int = 10000 #~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 50
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None
    # Network parameters
    encoder_type: str = 'Siren' #   
    n_frequencies: int = 352  #for classic, n_out = 2 * n_freq. For tcnn, n_out = 2 * n_freq * dim_in
    n_frequencies_t: int = 8 if encoder_type == 'tcnn' else 16
    dim_in: int = len(image_shape)
    dim_hidden: int = 256 
    dim_out: int = 1
    num_layers: int = 3
    skip_connections: tuple = () #(5, 11,)
    lr: float = 1e-4  # G requires training with a custom lr, usually lr * 0.1 
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
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=30.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        self.freqs = torch.nn.Linear(in_features, out_features, bias=bias)
        self.scale = torch.nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))
    
class ComplexGaborLayer(torch.nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = torch.nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = torch.nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = torch.nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())

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
        n_frequencies_t,
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
        self.n_frequencies_t = n_frequencies_t
        self.encoder_type = encoder_type

        # # self.encoder = tcnn.Encoding(n_input_dims=dim_in, encoding_config=config['encoding'])
        # if self.encoder_type == 'tcnn': #if tcnn is especially required, set it TODO: getattr more elegant
        #     #create the dictionary
        #     self.encoder = tcnn.Encoding(n_input_dims=(self.dim_in - 1), encoding_config={'otype': 'Frequency', 'n_frequencies': self.n_frequencies})
        #     self.encoder_t = tcnn.Encoding(n_input_dims=1, encoding_config={'otype': 'Frequency', 'n_frequencies': self.n_frequencies_t})
        # else: #fallback to classic
        #     self.encoder = rff.layers.GaussianEncoding(sigma=10.0, input_size=(self.dim_in - 1), encoded_size=self.n_frequencies)
        #     self.encoder_t = rff.layers.GaussianEncoding(sigma=10.0, input_size=1, encoded_size=self.n_frequencies_t)
            
        self.encoder = torch.nn.Sequential(Siren(dim_in=self.dim_in, dim_out=self.n_frequencies, is_first=True) , Siren(dim_in=self.n_frequencies, dim_out=self.n_frequencies), Siren(dim_in=self.n_frequencies, dim_out=1))
        # self.encoder = torch.nn.Sequential(RealGaborLayer(in_features=self.dim_in, out_features=self.n_frequencies, is_first=True) , RealGaborLayer(in_features=self.n_frequencies, out_features=self.n_frequencies), RealGaborLayer(in_features=self.n_frequencies, out_features=1))
        # self.encoder = torch.nn.Sequential(ComplexGaborLayer(in_features=self.dim_in, out_features=self.n_frequencies, is_first=True) , ComplexGaborLayer(in_features=self.n_frequencies, out_features=self.n_frequencies), ComplexGaborLayer(in_features=self.n_frequencies, out_features=1))
        self.encoding_dim_out = self.n_frequencies
        self.decoder = torch.nn.ModuleList()
        # self.decoder.append(Siren(dim_in=self.n_frequencies, dim_out=self.dim_out))
        # for i in range(self.n_layers):
        #     if i == 0:
        #         in_features = self.encoding_dim_out
        #     elif i in self.skip_connections:
        #         in_features = self.encoding_dim_out + self.dim_hidden
        #     else:
        #         in_features = self.dim_hidden
        #     block = torch.nn.Sequential(
        #         torch.nn.Linear(in_features=in_features, out_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden),
        #         torch.nn.BatchNorm1d(num_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden), #you can do batochnorm 3D + 1D and cat after
        #         torch.nn.GELU()
        #         # torch.nn.Tanh()
        #         # torch.nn.SiLU() #almost GeLU like
        #     )
        #     self.decoder.append(block)
            

    def forward(self, x):
        # coords = x[:, :(self.dim_in - 1)]
        # t = x[:, -1].unsqueeze(-1)
        # x = torch.hstack((self.encoder(coords), self.encoder_t(t))
        x = self.encoder(x)
        # # x = (x - x.min()) / (x.max() - x.min())
        # skip = x.clone()
        # for idx, layer in enumerate(self.decoder):
        #     if idx in self.skip_connections:
        #         x = torch.hstack((skip, x))
        #     x = layer(x)
        return x 

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) #weight_decay=1e-5
        return self.optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        loss = F.mse_loss(y_pred, y)

        self.log("train_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        return y_pred
    
    def lr_schedulers(self) -> LRSchedulerTypeUnion | List[LRSchedulerTypeUnion] | None:
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=10, verbose=True)
        return self.scheduler
    
    # def on_train_end(self) -> None:
    #     writer = SummaryWriter(log_dir=self.logger.log_dir)
    #     writer.add_text(text_string=str(config), tag='configuration')
    #     writer.close()
    #     # print(str(model.lr_schedulers().get_last_lr()))
        



mri_image = nib.load(config.image_path)

data = mri_image.get_fdata(dtype=np.float32)
data = data[:,:,3,7]
# data = Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))(torch.FloatTensor(data))

config.dim_in = len(data.shape)
config.image_shape = data.shape
# config.batch_size = int(np.prod(data.shape))

if config.checkpoint_path:
    model = FreqMLP.load_from_checkpoint(
                    checkpoint_path=config.checkpoint_path,
                    dim_in=config.dim_in, 
                    dim_hidden=config.dim_hidden, 
                    dim_out=config.dim_out, 
                    n_layers=config.num_layers, 
                    skip_connections=config.skip_connections,
                    n_frequencies=config.n_frequencies,
                    n_frequencies_t=config.n_frequencies_t,                
                    encoder_type=config.encoder_type,
                    lr=config.lr)
else:
    model = FreqMLP(dim_in=config.dim_in, 
                    dim_hidden=config.dim_hidden, 
                    dim_out=config.dim_out, 
                    n_layers=config.num_layers, 
                    skip_connections=config.skip_connections,
                    n_frequencies=config.n_frequencies,
                    n_frequencies_t=config.n_frequencies_t,                
                    encoder_type=config.encoder_type,
                    lr=config.lr)

Y = torch.FloatTensor(data).reshape(-1, 1)
Y = (Y - Y.min()) / (Y.max() - Y.min()) #* 2 - 1
# Y = torch.nn.functional.normalize(Y, p=2.0, dim=1, eps=1e-12, out=None)

axes = []
# for s in mri_image.shape:
for s in config.image_shape:
    axes.append(torch.linspace(0, 1, s))

mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

coords = torch.FloatTensor(mgrid)
X = coords.reshape(len(Y), config.dim_in)
            
dataset = torch.utils.data.TensorDataset(X, Y)

test_dataset = torch.utils.data.TensorDataset(X, Y)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
   
trainer = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
    precision=32,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)

# model = Siren2(in_features=config.dim_in, hidden_features=config.n_frequencies, hidden_layers=config.num_layers, out_features=config.dim_out)

# optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

#pretraining
trainer.fit(model, train_loader)

# for epoch in range(config.epochs):

#   # TRAINING LOOP
#   for train_batch in train_loader:
#     x, y = train_batch

#     y_pred, coords = model(x)
#     loss = F.mse_loss(y_pred, y)
#     print('train loss: ', loss.item())

#     loss.backward()

#     optimizer.step()
#     optimizer.zero_grad()

#   # VALIDATION LOOP
#   with torch.no_grad():
#     val_loss = []
#     for val_batch in mnist_val:
#       x, y = val_batch
#       logits = pytorch_model(x)
#       val_loss.append(cross_entropy_loss(logits, y).item())

#     val_loss = torch.mean(torch.tensor(val_loss))
#     print('val_loss: ', val_loss.item())

# #dense grid
# Y_interp = torch.zeros((np.prod(mri_image.shape) * config.interp_factor, 1))
# axes = []
# for idx, s in enumerate(mri_image.shape):
#     if idx == (len(mri_image.shape) - 1):
#         axes.append(torch.linspace(0, 1, s * config.interp_factor))        
#     else:
#         axes.append(torch.linspace(0, 1, s))
        
# mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

# coords = torch.FloatTensor(mgrid)
# X_interp = coords.reshape(len(Y_interp), config.dim_in)    

# interp_dataset = torch.utils.data.TensorDataset(X_interp, Y_interp)
# interp_loader = torch.utils.data.DataLoader(interp_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
try:
    filepath = model.logger.log_dir + '/'
except:
    count = int(len(os.listdir('lightning_logs')))
    os.mkdir(f'lightning_logs/version_{count}')
    filepath = f'lightning_logs/version_{(count)}/'

# #create a prediction
# pred = torch.zeros(1, 1)
# with torch.no_grad():
#     for batch in test_loader:
#         pred = torch.concat((pred, model(batch[0])[0]))
# pred = pred[1:]          

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
# nib.save(nib.Nifti1Image(im, affine=np.eye(4)), filepath + 'pred_before_reg.nii.gz')

# #create an interpolation
# interp = torch.concat(trainer.predict(model, interp_loader))
            
# interp_im = interp.reshape((mri_image.shape[0], mri_image.shape[1], mri_image.shape[2], mri_image.shape[3] * config.interp_factor))
# interp_im = interp_im.detach().cpu().numpy()
# interp_im = np.array(interp_im, dtype=np.float32)
# nib.save(nib.Nifti1Image(interp_im, affine=np.eye(4)), filepath + 'interpolation.nii.gz')

config.export_to_txt(file_path=filepath)
