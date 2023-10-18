import importlib
import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, List, Tuple, Union

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch

from datamodules import MNISTDataModule, MriDataModule, MriFramesDataModule
from models import (HashMLP, HashSirenNet, ModulatedSirenNet, MultiHashMLP,
                    MultiSiren, SirenNet)


@dataclass
class BaseConfig:
    checkpoint_path = None
    # image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_path: str = 'sample_ankle_dyn_mri.nii.gz'
    image_shape = nib.load(image_path).shape
    batch_size: int = 4096  # int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 1
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = (
        None  # MappingProxyType({200: 2}) #MappingProxyType({0: 5})
    )
    # image_path: str = 'data/equinus_frames/frame8.nii.gz'
    # image_path: str = '/mnt/Data/DHCP/sub-CC00074XX09_ses-28000_desc-restore_T2w.nii.gz'
    # image_path:str = '/mnt/Data/HCP/HCP100_T1T2/146432_T2.nii.gz'

    # image_path: str = '/home/aorus-users/Benjamin/git_repos/mri_interpolation/data/equinus_sameframes.nii.gz'
    # image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    # image_path: str = 'data/equinus_singleframe_noisy.nii.gz'

    # Network parameters
    dim_in: int = len(image_shape)
    dim_hidden: int = 128
    dim_out: int = 1
    n_layers: int = 6
    n_sample: int = 3
    w0: float = 30.0
    w0_initial: float = 30.0
    use_bias: bool = True
    final_activation = None
    lr: float = 1e-4  # G requires training with a custom lr, usually lr * 0.1
    datamodule: pl.LightningDataModule = MriDataModule
    model_cls: pl.LightningModule = HashMLP

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")
                
@dataclass
class HashConfig(BaseConfig):
    checkpoint_path: str = None
    image_path: str = 'sample_ankle_dyn_mri.nii.gz'
    image_shape = nib.load(image_path).shape
    interp_shapes = [(352, 352, 30)]
    batch_size: int = 20000  # ~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 1
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None
    encoder_type: str = "hash"  
    # Network parameters
    n_levels: int = 8
    n_features_per_level: int = 4
    log2_hashmap_size: int = 23
    base_resolution: MappingProxyType = (64, 64, 5)
    finest_resolution: MappingProxyType = (352, 352, 15)
    # base_resolution: int = 64
    # finest_resolution: int = 512
    per_level_scale: int = 1.2
    interpolation: str = "Linear"  # can be "Nearest", "Linear" or "Smoothstep", not used if not 'tcnn' encoder_type
    dim_in: int = len(image_shape)
    dim_hidden: int = 64
    dim_out: int = 1
    n_layers: int = 2
    lr: float = 5e-3  
    dropout: float = 0.0

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")


# @dataclass
# class MNISTConfig:
#     batch_size: int = (
#         784  # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST
#     )
#     inner_loop_it: int = 5
#     outer_loop_it: int = 10
#     epochs: int = 1
#     num_workers: int = os.cpu_count()
#     device = [0] if torch.cuda.is_available() else []
#     fixed_seed: bool = True
#     # dataset_path: str = '/home/benjamin/Documents/Datasets' #for MNIST
#     dataset_path: str = "mnt/Data/"
#     # image_path: str = '/home/benjamin/Documents/Datasets/HCP/100307_T2.nii.gz'
#     image_path: str = "data/t2_256cube.nii.gz"
#     train_target: tuple = (2,)
#     test_target: tuple = (7,)
#     initialization: str = "single"
#     apply_psf: bool = False
#     hashconfig_path: str = "hash_config.json"

#     # Network parameters
#     dim_in: int = 2
#     dim_hidden: int = 256
#     dim_out: int = 1
#     n_layers: int = 5
#     w0: float = 1.0
#     w0_initial: float = 30.0
#     use_bias: bool = True
#     final_activation = None
#     lr: float = 1e-3  # G requires training with a custom lr, usually lr * 0.1
#     opt_type: str = "LSTM"
#     conv_channels: tuple = (
#         8,
#         8,
#         8,
#     )
#     datamodule: pl.LightningDataModule = MNISTDataModule
#     accum: MappingProxyType = MappingProxyType({200: 3, 400: 4})

#     # output
#     output_path: str = "results_fourrier/"
#     if os.path.isdir(output_path) is False:
#         os.mkdir(output_path)
#     experiment_number: int = (
#         0 if len(os.listdir(output_path)) == 0 else len(os.listdir(output_path))
#     )

#     def export_to_txt(self, file_path: str = "") -> None:
#         with open(file_path + "config.txt", "w") as f:
#             for key in self.__dict__:
#                 f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")


def string_to_class(string):
    """
    String to class protoype for json or yaml style configuration
    """
    string_list = string.split(".")
    cls_string = str(string_list[-1])
    module_string = str(string_list[:-1])

    # needs importlib
    MyClass = getattr(importlib.import_module(module_string), cls_string)
    return MyClass
