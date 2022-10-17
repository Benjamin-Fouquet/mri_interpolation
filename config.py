from dataclasses import dataclass, field
import os
import torch
from typing import Union, Tuple, Set
import pytorch_lightning as pl

from datamodules import MriDataModule, MNISTDataModule

@dataclass
class Config:
    batch_size: int = 300000 #28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST
    inner_loop_it: int = 5
    outer_loop_it: int = 10
    epochs: int = 20
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    fixed_seed: bool = True
    dataset_path: str = '/home/benjamin/Documents/Datasets'
    # image_path: str = '/home/benjamin/Documents/Datasets/HCP/100307_T2.nii.gz'
    image_path:str = '/mnt/Data/HCP/HCP100_T1T2/100307_T2.nii.gz'
    train_target: Tuple = (2,)
    test_target: Tuple = (7,)
    initialization: str = 'single'
    apply_psf: bool = False

    #Network parameters
    dim_in: int = 3
    dim_hidden: int = 512
    dim_out:int = 1
    num_layers:int = 5
    w0: float = 30.0
    w0_initial:float = 30.0
    use_bias: bool = True
    final_activation = None
    lr: float = 1e-4 #G requires training with a custom lr, usually lr * 0.1
    opt_type: str = 'LSTM'
    conv_channels: Tuple = (8, 8, 8,)
    datamodule: pl.LightningDataModule = MriDataModule

    comment: str = 'Mean initialisation'

    #output
    output_path:str = 'results_siren/'
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path)
    experiment_number:int = 0 if len(os.listdir(output_path)) == 0 else len(os.listdir(output_path))

    def export_to_txt(self, file_path: str = '') -> None:
        with open(file_path + 'config.txt', 'w') as f:
            for key in self.__dict__:
                f.write(str(key) + ' : ' + str(self.__dict__[key]) + '\n')


if __name__ == 'main':
    #test initialization
    config = Config()