'''
hyperparameter search for siren using optuna
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
import optuna
import logging
import sys

torch.manual_seed(1337)

@dataclass
class BaseConfig:
    checkpoint_path = None #'lightning_logs/version_25/checkpoints/epoch=49-step=11200.ckpt'
    log: str = None
    # image_path: str = '/mnt/Data/FetalAtlas/template_T2.nii.gz'
    image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_shape = nib.load(image_path).shape
    batch_size: int = 250000 #~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 30
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None 
    # Network parameters
    encoder_type: str = 'siren' #   
    n_frequencies: int = 32 if encoder_type == 'tcnn' else 352 #for classic, n_out = 2 * n_freq. For tcnn, n_out = 2 * n_freq * dim_in
    sigma: float = 20.0
    w0: float = 30.0
    n_frequencies_t: int = 4 if encoder_type == 'tcnn' else 30
    sigma_t: float = 2.0
    w0_t: float = 30.0
    dim_in: int = len(image_shape)
    dim_hidden: int = 64 
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


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

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
        sigma,
        w0,
        n_frequencies_t,
        w0_t,
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
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.w0 = w0
        self.n_frequencies_t = n_frequencies_t
        self.w0_t = w0_t
        self.sigma_t = sigma_t
        self.encoder_type = encoder_type
        self.second_training = False

        # self.encoder = tcnn.Encoding(n_input_dims=dim_in, encoding_config=config['encoding'])
        if self.encoder_type == 'siren':
            self.encoder = torch.nn.Sequential(Siren(dim_in=(self.dim_in - 1),dim_out=self.n_frequencies, is_first=True, w0=self.w0, c=self.sigma), Siren(dim_in=self.n_frequencies ,dim_out=self.n_frequencies, is_first=False, w0=self.w0, c=self.sigma))
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
        if isinstance(self.encoder, GaussianFourierFeatureTransform):
            coords = coords.unsqueeze(-1).unsqueeze(-1)
            t = t.unsqueeze(-1).unsqueeze(-1)
        x = torch.hstack((self.encoder(coords), self.encoder_t(t)))
        skip = x.clone()
        if isinstance(self.encoder, GaussianFourierFeatureTransform):
            x = x.squeeze(-1).squeeze(-1)
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
        self.final_loss = float(loss.detach().cpu().numpy()) #parameter used for optuna
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
data = data[:,:,:,:] #optional line for doing 3D and accelerate prototyping
config.image_shape = data.shape
config.dim_in = len(data.shape)
 
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

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "stacked_siren_study"  # Unique identifier of the study.
storage_name = f"sqlite:///{study_name}.db"

def objective(trial):
    #parameters to search
    config.n_frequencies = trial.suggest_int("n_frequencies", 64, 256)
    config.sigma = trial.suggest_float("sigma", 2, 30)
    config.w0 = trial.suggest_float("w0", 1.0, 50.0)
    config.n_frequencies_t = trial.suggest_int("n_frequencies_t", 5, 30)
    config.sigma_t = trial.suggest_float("sigma_t", 1, 10)
    config.w0_t = trial.suggest_float("w0_t", 1.0, 50.0)
    config.num_layers = trial.suggest_int("num_layers", 3, 10)
    config.dim_hidden = trial.suggest_int('dim_hidden', 32, 512)
    
    model = FreqMLP(dim_in=config.dim_in, 
                    dim_hidden=config.dim_hidden, 
                    dim_out=config.dim_out, 
                    n_layers=config.num_layers, 
                    skip_connections=config.skip_connections,
                    n_frequencies=config.n_frequencies,
                    sigma=config.sigma,
                    w0=config.w0,
                    n_frequencies_t=config.n_frequencies_t,
                    sigma_t=config.sigma_t,
                    w0_t=config.w0_t,                
                    encoder_type=config.encoder_type,
                    lr=config.lr) 
    
   
    trainer = pl.Trainer(
        gpus=config.device,
        max_epochs=config.epochs,
        accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
        precision=32,
        # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
    )

    trainer.fit(model, train_loader)
    
    return model.final_loss

study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=1)

filepath = 'optuna_studies/'

with open(filepath + 'best_params_stacked_sirenMLP.txt', 'w') as f:
    print(study.best_params, file=f)

study.trials_dataframe().to_csv(filepath + 'stacked_sirenMLP_tests.csv')





            



