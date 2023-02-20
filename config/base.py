

import pytorch_lightning as pl
import torch
import os
import nibabel as nib
from datamodules import MriDataModule, MNISTDataModule, MriFramesDataModule
from models import HashSirenNet, SirenNet, ModulatedSirenNet, HashMLP, MultiHashMLP, MultiSiren
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union, FrozenSet, List
from types import MappingProxyType
import numpy as np
import importlib

@dataclass
class BaseConfig:
    checkpoint_path = None
    batch_size: int = 743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 300
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None #MappingProxyType({200: 2}) #MappingProxyType({0: 5})
    image_path: str = 'data/equinus_frames/frame8.nii.gz'
    # image_path: str = 'data/equinus_downsampled.nii.gz'
    # image_path: str = '/home/aorus-users/Benjamin/git_repos/mri_interpolation/data/equinus_sameframes.nii.gz'
    # image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    # image_path: str = 'data/equinus_singleframe_noisy.nii.gz'
    image_shape = nib.load(image_path).shape
    coordinates_spacing: np.array = np.array(
        (2 / image_shape[0], 2 / image_shape[1], 2 / image_shape[2])
    )
    hashconfig_path: str = 'config/hash_config.json'

    # Network parameters
    dim_in: int = len(image_shape)
    dim_hidden: int = 64
    dim_out: int = 1
    num_layers: int = 6
    n_sample: int = 3
    w0: float = 30.0
    w0_initial: float = 30.0
    use_bias: bool = True
    final_activation = None
    lr: float = 1e-3  # G requires training with a custom lr, usually lr * 0.1
    datamodule: pl.LightningDataModule = MriDataModule
    model_cls: pl.LightningModule = HashMLP  
    # datamodule: pl.LightningDataModule = MriFramesDataModule
    # model_cls: pl.LightningModule = MultiHashMLP  
    n_frames: int = 15

    # # output
    # output_path: str = "results_hash/"
    # if os.path.isdir(output_path) is False:
    #     os.mkdir(output_path)
    # experiment_number: int = 0 if len(os.listdir(output_path)) == 0 else len(
    #     os.listdir(output_path)
    # )

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")

@dataclass
class MNISTConfig:
    batch_size: int = 784  # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST
    inner_loop_it: int = 5
    outer_loop_it: int = 10
    epochs: int = 1
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    fixed_seed: bool = True
    # dataset_path: str = '/home/benjamin/Documents/Datasets' #for MNIST
    dataset_path: str = "mnt/Data/"
    # image_path: str = '/home/benjamin/Documents/Datasets/HCP/100307_T2.nii.gz'
    image_path: str = "data/t2_256cube.nii.gz"
    train_target: tuple = (2,)
    test_target: tuple = (7,)
    initialization: str = "single"
    apply_psf: bool = False
    hashconfig_path: str = 'hash_config.json'

    # Network parameters
    dim_in: int = 2
    dim_hidden: int = 256
    dim_out: int = 1
    num_layers: int = 5
    w0: float = 1.0
    w0_initial: float = 30.0
    use_bias: bool = True
    final_activation = None
    lr: float = 1e-3  # G requires training with a custom lr, usually lr * 0.1
    opt_type: str = "LSTM"
    conv_channels: tuple = (8, 8, 8,)
    datamodule: pl.LightningDataModule = MNISTDataModule
    accum: MappingProxyType = MappingProxyType({200: 3, 400: 4})

    # output
    output_path: str = "results_fourrier/"
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path)
    experiment_number: int = 0 if len(os.listdir(output_path)) == 0 else len(
        os.listdir(output_path)
    )

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")

def string_to_class(string):
    '''
    String to class protoype for json or yaml style configuration
    '''
    string_list = string.split('.')
    cls_string = str(string_list[-1])
    module_string = str(string_list[:-1])

    #needs importlib
    MyClass = getattr(importlib.import_module(module_string), cls_string)
    return MyClass
