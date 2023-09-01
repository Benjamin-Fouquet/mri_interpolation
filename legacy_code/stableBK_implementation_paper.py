'''
Implementation 'Continuous Longitudinal Fetus Brain Atlas Construction via Implicit Neural Representation' using tinycuda

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

torch.manual_seed(1337)

@dataclass
class BaseConfig:
    checkpoint_path = None #'lightning_logs/version_384/checkpoints/epoch=99-step=100.ckpt'
    image_path: str = '/mnt/Data/FetalAtlas/template_T2.nii.gz'
    # image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_shape = nib.load(image_path).shape
    batch_size: int = 200000 #~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 10
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None 
    # Network parameters
    encoder_type: str = 'tcnn' #   
    n_frequencies: int = 64 if encoder_type == 'tcnn' else 128  #for classic, n_out = 2 * n_freq. For tcnn, n_out = 2 * n_freq * dim_in
    n_frequencies_t: int = 8 if encoder_type == 'tcnn' else 16
    dim_in: int = len(image_shape)
    dim_hidden: int = 256 
    dim_out: int = 1
    num_layers: int = 18
    skip_connections: tuple = ()#(5, 11,)
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

        # self.encoder = tcnn.Encoding(n_input_dims=dim_in, encoding_config=config['encoding'])
        if self.encoder_type == 'tcnn': #if tcnn is especially required, set it TODO: getattr more elegant
            #create the dictionary
            self.encoder = tcnn.Encoding(n_input_dims=(self.dim_in - 1), encoding_config={'otype': 'Frequency', 'n_frequencies': self.n_frequencies})
            self.encoder_t = tcnn.Encoding(n_input_dims=1, encoding_config={'otype': 'Frequency', 'n_frequencies': self.n_frequencies_t})
        else: #fallback to classic
            self.encoder = rff.layers.GaussianEncoding(sigma=10.0, input_size=(self.dim_in - 1), encoded_size=self.n_frequencies)
            self.encoder_t = rff.layers.GaussianEncoding(sigma=10.0, input_size=1, encoded_size=self.n_frequencies_t)
            
        self.encoding_dim_out = (self.n_frequencies * 2 * (self.dim_in - 1) + self.n_frequencies_t * 2) if isinstance(self.encoder, tcnn.Encoding) else (self.n_frequencies * 2 + self.n_frequencies_t * 2)
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
                # torch.nn.GELU()
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

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5) #weight_decay=1e-5
        return self.optimizer

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
    
    # def lr_schedulers(self) -> LRSchedulerTypeUnion | List[LRSchedulerTypeUnion] | None:
    #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=10, verbose=True)
    #     return self.scheduler
    
    # def on_train_end(self) -> None:
    #     writer = SummaryWriter(log_dir=self.logger.log_dir)
    #     writer.add_text(text_string=str(config), tag='configuration')
    #     writer.close()
    #     # print(str(model.lr_schedulers().get_last_lr()))
        


model_1 = FreqMLP(dim_in=config.dim_in, 
                dim_hidden=config.dim_hidden, 
                dim_out=config.dim_out, 
                n_layers=config.num_layers, 
                skip_connections=config.skip_connections,
                n_frequencies=config.n_frequencies,
                n_frequencies_t=config.n_frequencies_t,                
                encoder_type=config.encoder_type,
                lr=config.lr)

# model_2 = FreqMLP(dim_in=config.dim_in, 
#                 dim_hidden=config.dim_hidden, 
#                 dim_out=config.dim_out, 
#                 n_layers=config.num_layers, 
#                 skip_connections=config.skip_connections,
#                 n_frequencies=config.n_frequencies,
#                 n_frequencies_t=config.n_frequencies_t,                
#                 encoder_type=config.encoder_type,
#                 lr=config.lr)

mri_image = nib.load(config.image_path)

data = mri_image.get_fdata(dtype=np.float32)
data_1 = data[:,:,:,::2].copy()
data_2 = data[:,:,:,1::2].copy()

Y = torch.FloatTensor(data).reshape(-1, 1)
Y_1 = torch.FloatTensor(data_1).reshape(-1, 1)
Y_1 = Y_1 / Y_1.max()
Y_2 = torch.FloatTensor(data_2).reshape(-1, 1)
Y_2 = Y_2 / Y_2.max()

axes = []
for s in mri_image.shape:
    axes.append(torch.linspace(0, 1, s))

mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

coords = torch.FloatTensor(mgrid)
X = coords.reshape(len(Y), config.dim_in)
X_1 = coords[:,:,:,::2,:].reshape(len(Y_1), config.dim_in)
X_2 = coords[:,:,:,1::2,:].reshape(len(Y_2), config.dim_in)

dataset = torch.utils.data.TensorDataset(X, Y)
dataset_1 = torch.utils.data.TensorDataset(X_1, Y_1)
dataset_2 = torch.utils.data.TensorDataset(X_2, Y_2)

train_loader_1 = torch.utils.data.DataLoader(dataset_1, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
train_loader_2 = torch.utils.data.DataLoader(dataset_2, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
test_loader_1 = torch.utils.data.DataLoader(dataset_1, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)  
test_loader_2 = torch.utils.data.DataLoader(dataset_2, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)  
   
trainer_1 = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
    precision=16,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)
# trainer_2 = pl.Trainer(
#     gpus=config.device,
#     max_epochs=config.epochs,
#     accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
#     precision=16,
#     # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
# )

#pretraining
trainer_1.fit(model_1, train_loader_1)
# trainer_2.fit(model_2, train_loader_2)

#refine stage

#dense grid
Y_interp = torch.zeros((np.prod(mri_image.shape) * config.interp_factor, 1))
axes = []
for idx, s in enumerate(mri_image.shape):
    if idx == (len(mri_image.shape) - 1):
        axes.append(torch.linspace(0, 1, s * config.interp_factor))        
    else:
        axes.append(torch.linspace(0, 1, s))
        
mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

coords = torch.FloatTensor(mgrid)
X_interp = coords.reshape(len(Y_interp), config.dim_in)    

interp_dataset = torch.utils.data.TensorDataset(X_interp, Y_interp)
interp_loader = torch.utils.data.DataLoader(interp_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

filepath = model_1.logger.log_dir + '/'

#create a prediction
pred_1 = torch.concat(trainer_1.predict(model_1, test_loader))
# pred_2 = torch.concat(trainer_2.predict(model_2, test_loader))

# pred = (pred_1 + pred_2 ) / 2
pred = pred_1
            
im = pred.reshape(mri_image.shape)
im = im.detach().cpu().numpy()
im = np.array(im, dtype=np.float32)
nib.save(nib.Nifti1Image(im, affine=np.eye(4)), filepath + 'pred_before_reg.nii.gz')

#create an interpolation
interp_1 = torch.concat(trainer_1.predict(model_1, interp_loader))
# interp_2 = torch.concat(trainer_2.predict(model_2, interp_loader))
# interp = (interp_1 + interp_2) / 2

interp = interp_1
            
interp_im = interp.reshape((mri_image.shape[0], mri_image.shape[1], mri_image.shape[2], mri_image.shape[3] * config.interp_factor))
interp_im = interp_im.detach().cpu().numpy()
interp_im = np.array(interp_im, dtype=np.float32)
nib.save(nib.Nifti1Image(interp_im, affine=np.eye(4)), filepath + 'interpolation.nii.gz')


# #training loop for refine stage, old pytorch style
# optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=config.lr)
# optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=config.lr)


# if config.device == [0]:
#     model_1 = model_1.to('cuda')
#     model_2 = model_2.to('cuda')

# for _ in range(config.epochs):
#     for train_batch in interp_loader:
#         x, y = train_batch
#         if config.device == [0]:
#             x = x.to('cuda')
#             y = y.to('cuda')
        
#         pred_1 = model_1(x)
#         pred_2 = model_2(x)
        
#         loss = F.mse_loss(pred_1, pred_2)
#         print(str(loss))

#         loss.backward()

#         optimizer_1.step()
#         optimizer_1.zero_grad()
#         optimizer_2.step()
#         optimizer_2.zero_grad()
#         print(str(loss))

# if config.device == [0]:
#     model_1 = model_1.to('cpu')
#     model_2 = model_2.to('cpu')

# #create a prediction
# pred_1 = torch.concat(trainer_1.predict(model_1, test_loader))
# pred_2 = torch.concat(trainer_2.predict(model_2, test_loader))

# pred = (pred_1 + pred_2 ) / 2
            
# im = pred.reshape(mri_image.shape)
# im = im.detach().cpu().numpy()
# im = np.array(im, dtype=np.float32)
# nib.save(nib.Nifti1Image(im, affine=np.eye(4)), filepath + 'pred_after_reg.nii.gz')

# #t interp * 10
# Y_interp = torch.zeros((np.prod(mri_image.shape) * config.interp_factor, 1))

# axes = []
# #reordering to get time correctly
# new_shape = [mri_image.shape[-1]]
# new_shape.extend(mri_image.shape[0:3])
# for idx, s in enumerate(mri_image.shape):
#     if idx == 0:
#         axes.append(torch.linspace(0, 1, s * config.interp_factor))        
#     else:
#         axes.append(torch.linspace(0, 1, s))
        
# mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

# mgrid = mgrid.swapaxes(0, 3) #put time to last axe

# coords = torch.FloatTensor(mgrid)
# X_interp = coords.reshape(len(Y_interp), config.dim_in)    

# interp_dataset = torch.utils.data.TensorDataset(X_interp, Y_interp)
# interp_loader = torch.utils.data.DataLoader(interp_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

# #create an interpolation
# interp_1 = torch.concat(trainer_1.predict(model_1, interp_loader))
# interp_2 = torch.concat(trainer_2.predict(model_2, interp_loader))
# interp = (interp_1 + interp_2) / 2
            
# interp_im = interp.reshape((mri_image.shape[0], mri_image.shape[1], mri_image.shape[2], mri_image.shape[3] * config.interp_factor))
# interp_im = interp_im.detach().cpu().numpy()
# interp_im = np.array(interp_im, dtype=np.float32)
# nib.save(nib.Nifti1Image(interp_im, affine=np.eye(4)), filepath + 'interpolation.nii.gz')

config.export_to_txt(file_path=filepath)

