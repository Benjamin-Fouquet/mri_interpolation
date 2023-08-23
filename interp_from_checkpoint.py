from typing import List, Optional, Union
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
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
from skimage import metrics

filepath = '/home/benjamin/results_repaper/version_6/'

@dataclass
class BaseConfig:
    checkpoint_path = '/home/benjamin/results_repaper/version_6/checkpoints/epoch=499-step=18000.ckpt'
    # image_path: str = '/mnt/Data/FetalAtlas/template_T2.nii.gz'
    image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_shape = nib.load(image_path).shape
    batch_size: int = 100000 #~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 100
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None 
    model_cls: pl.LightningModule = HashMLP
    encoder_type: str = 'hash' #   
    # Network parameters
    n_levels: int = 8
    n_features_per_level: int = 2
    log2_hashmap_size: int = 23
    base_resolution: MappingProxyType = (64, 64, 64, 4)
    finest_resolution: MappingProxyType = (512, 512, 512, 8)
    # base_resolution: int = 64
    # finest_resolution: int = 512
    per_level_scale: int = 1.5
    interpolation: str = "Linear" #can be "Nearest", "Linear" or "Smoothstep", not used if not 'tcnn' encoder_type
    dim_in: int = len(image_shape)
    dim_hidden: int = 128 
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

model= config.model_cls.load_from_checkpoint(
                config.checkpoint_path, 
                dim_in=config.dim_in,
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

interp_shape = (350, 350, 6, 15)

axes = []
for idx, s in enumerate(interp_shape):
    axes.append(torch.linspace(0, 1, s))
    
        
mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

coords = torch.FloatTensor(mgrid)

Y_interp = torch.zeros(np.prod(interp_shape), 1)

X_interp = coords.reshape(len(Y_interp), len(interp_shape))    

interp_dataset = torch.utils.data.TensorDataset(X_interp, Y_interp)
interp_loader = torch.utils.data.DataLoader(interp_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
   
trainer = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
    precision=32,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)

#create an interpolation
interp = torch.concat(trainer.predict(model, interp_loader))
   
interp_im = interp.reshape(interp_shape)

interp_im = interp_im.detach().cpu().numpy()
interp_im = np.array(interp_im, dtype=np.float32)
nib.save(nib.Nifti1Image(interp_im, affine=np.eye(4)), filepath + f'interpolation{str(interp_shape)}.nii.gz')