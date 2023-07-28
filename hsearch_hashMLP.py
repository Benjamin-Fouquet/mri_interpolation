'''
Hyperparameter search for HashMLP using optuna
'''
from typing import List, Optional, Union
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
import tinycudann as tcnn 
import torch
import pytorch_lightning as pl 
import torch.nn.functional as F
import nibabel as nib 
from dataclasses import dataclass
import os
from types import MappingProxyType
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from math import pi
import torch.utils.data
import matplotlib.pyplot as plt
import encoding
from skimage import metrics
import optuna
import logging
import sys
from scipy.fft import fftn, ifftn, fftshift

torch.manual_seed(1337)

@dataclass
class BaseConfig:
    checkpoint_path = None #'lightning_logs/version_384/checkpoints/epoch=99-step=100.ckpt'
    # image_path: str = '/mnt/Data/FetalAtlas/template_T2.nii.gz'
    image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_shape = nib.load(image_path).shape
    batch_size: int = 250000 #~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 50
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None 
    encoder_type: str = 'hash' #   
    # Network parameters
    n_levels: int = 16
    n_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: MappingProxyType = (64, 64, 4,  4)
    finest_resolution: MappingProxyType = (512, 512, 12, 30)
    # base_resolution: int = 64
    # finest_resolution: int = 512
    per_level_scale: int = 1.5
    interpolation: str = "Linear" #can be "Nearest", "Linear" or "Smoothstep", not used if not 'tcnn' encoder_type
    dim_in: int = len(image_shape)
    dim_hidden: int = 64 
    dim_out: int = 1
    num_layers: int = 4
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


class HashMLP(pl.LightningModule):
    '''
    Lightning module for HashMLP. 
    '''
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        n_layers,
        encoder_type,
        n_levels,
        n_features_per_level,
        log2_hashmap_size,
        base_resolution,
        finest_resolution,
        per_level_scale,
        interpolation,
        lr,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.encoder_type = encoder_type
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.per_level_scale = per_level_scale
        self.interpolation = interpolation
        self.lr = lr

        if self.encoder_type == 'tcnn':
            self.encoder = torch.nn.utils.parametrizations.spectral_norm(tcnn.Encoding(n_input_dims=(self.dim_in), encoding_config= {"otype": "HashGrid", "n_levels": self.n_levels, "n_features_per_level": self.n_features_per_level, "log2_hashmap_size": self.log2_hashmap_size, "base_resolution": self.base_resolution, "per_level_scale": self.per_level_scale, "interpolation": self.interpolation}, dtype=torch.float32), name='params', n_power_iterations=4, eps=1e-12, dim=None)
        
        else: 
            if isinstance(self.base_resolution, int):
                self.encoder = encoding.MultiResHashGrid(
                    dim=self.dim_in, 
                    n_levels=self.n_levels, 
                    n_features_per_level=self.n_features_per_level,
                    log2_hashmap_size=self.log2_hashmap_size,
                    base_resolution=self.base_resolution,
                    finest_resolution=self.finest_resolution,
                    )
            else:
                self.encoder = encoding.MultiResHashGridV2(
                    dim=self.dim_in, 
                    n_levels=self.n_levels, 
                    n_features_per_level=self.n_features_per_level,
                    log2_hashmap_size=self.log2_hashmap_size,
                    base_resolution=self.base_resolution,
                    finest_resolution=self.finest_resolution,
                    )
        
        self.encoding_dim_out = self.n_levels * self.n_features_per_level

        self.decoder = torch.nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                in_features = self.encoding_dim_out
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
        x = self.encoder(x)
        for idx, layer in enumerate(self.decoder):
            x = layer(x)
        return x

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4) #weight_decay=1e-5
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
study_name = "hash_study"  # Unique identifier of the study.
storage_name = f"sqlite:///{study_name}.db"

def objective(trial):
    #parameters to search
    config.num_layers = trial.suggest_int("num_layers", 3, 10)
    config.dim_hidden = trial.suggest_int('dim_hidden', 64, 256)
    br_slice = trial.suggest_int('starting_slice_resolution', 2, 6)
    br_time = trial.suggest_int('starting_time_resolution', 4, 15)
    fr_slice = trial.suggest_int('final_slice_resolution', 6, 6)
    fr_time = trial.suggest_int('final_time_resolution', 15, 15)
    config.base_resolution = (64, 64, br_slice, br_time)
    config.finest_resolution = (512, 512, fr_slice, fr_time)
    config.n_levels = trial.suggest_int('n_levels', 8, 32)
    config.n_features_per_level = trial.suggest_int('n_features_per_level', 1, 4)
    config.log2_hashmap_size = trial.suggest_int('log2_hashmap_size', 16, 24)
    
    model = HashMLP(dim_in=config.dim_in, 
                    dim_hidden=config.dim_hidden, 
                    dim_out=config.dim_out, 
                    n_layers=config.num_layers,
                    encoder_type=config.encoder_type,
                    n_levels=config.n_levels,
                    n_features_per_level=config.n_features_per_level,
                    log2_hashmap_size=config.log2_hashmap_size,
                    base_resolution=config.base_resolution,
                    finest_resolution=config.finest_resolution,
                    per_level_scale=config.per_level_scale,
                    interpolation=config.interpolation,
                    lr=config.lr)
    
   
    trainer = pl.Trainer(
        gpus=config.device,
        max_epochs=config.epochs,
        accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
        precision=32,
        # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
    )

    trainer.fit(model, train_loader)
    
    # #create a prediction
    # pred = torch.concat(trainer.predict(model, test_loader))
                
    # im = pred.reshape(config.image_shape)
    # im = im.detach().cpu().numpy()
    # im = np.array(im, dtype=np.float32)
    
    #do DFT on prediction
    # fourier = fftshift(fftn(im))    
    #quantify high freq components ?
    
    #return
    
    return model.final_loss

study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=10)

filepath = 'optuna_studies/'

with open(filepath + 'best_params_hashMLP.txt', 'w') as f:
    print(study.best_params, file=f)

study.trials_dataframe().to_csv(filepath + 'hashMLP_tests.csv')





            



