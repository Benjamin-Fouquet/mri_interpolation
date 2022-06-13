'''
TODO:

-try SeparableConv2D as drop in replacement for 2D conv
-checkpoints for easy save/load of model
-automated 10% patch overlap
-upscale baboon images before test, to match resolution of training
-make degradation flexible ?
-log for tensorboard at train and val
DONE:
-2D implementation (and 3D)
-define train, val, test datasets
-patch structure
-optimisation artefacts creation
-some logs
-2D patchs with adaptation and 2D convs !! not possible with torchio
'''

import glob
import multiprocessing
import sys
from typing import Tuple, Union, Dict
from matplotlib import transforms
from dataclasses import dataclass
import argparse

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
import torchio as tio
from torch.utils.data import DataLoader, Dataset

from models import AutoEncoder, ThreeDCNN, ConvModule, UnetAdded, UnetConcatenated

from utils import * #utils for debugging, to be removed at deployement

gpu = [0] if torch.cuda.is_available() else []

#placeholder dataclass for hydra config // can you set it up on the flight via command line easily ? or call alternative configuration on the fly
@dataclass
class Config:
    model_class: pl.LightningModule = AutoEncoder
    batch_size: int = 32
    max_subjects: int = 100
    dhcp_path: str = "data/DHCP_seg/"
    baboon_path: str ='data/baboon_seg/'
    patch_size: Union[int, Tuple[int, ...]] = (64, 64, 1)
    patch_overlap: Union[int, Tuple[int, ...]] = None
    max_epochs: int = 50
    num_channels: Tuple[int, ...] = (32, 64, 128, 64, 32)

    def export_to_txt(self, file_path: str = '') -> None:
        with open(file_path + 'config.txt', 'w') as f:
            for key in self.__dict__:
                f.write(str(key) + ' : ' + str(self.__dict__[key]) + '\n')

config = Config()

class MriDataModule(pl.LightningDataModule):
    '''
    MRI DataModule taking nifti images, applying torchio transforms to it and returning a batch of associated data. Used for training on DHCP and testing on baboon. 3D
    ARGS:
    -dhcp_path: path to training data folder
    -baboon_path: path to test data folder
    -max_subjects: max number of subjects for training data
    -batch_size: batch_size
    -patch_size: size of patch, None yield complete image
    -patch overlap: overlap between patches: for the moment hard coded 10%
    '''
    def __init__(
        self,
        dhcp_path: str = "data/DHCP_seg/",
        baboon_path: str ='data/baboon_seg/',
        max_subjects: int = 10, #max is 490
        batch_size: int = 32,
        patch_size: Union[int, Tuple[int, ...]] = (64, 64, 1),
        patch_overlap: Union[int, Tuple[int, ...]] = None, #no effect atm
        *args,
        **kwargs
    ):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.dhcp_path = dhcp_path
        self.batch_size = batch_size
        self.baboon_path = baboon_path
        self.max_subjects = max_subjects
        #train/val ~10% hardcoded // potential bug if max subj > number of available data
        self.train_length = int(self.max_subjects * 0.9)
        self.val_length = self.max_subjects - self.train_length
        self.patch_size = patch_size
        self.patch_overlap = (int(size / 10) for size in patch_size) #hardcoded

    def setup(self):
        all_t2s = glob.glob(self.dhcp_path + '*_t2_seg.nii.gz', recursive=True)[:self.max_subjects]
        subjects = []
        down_factor = 3.0
        # DHCP data for train and val
        for t2_path in all_t2s:
            #normalize
            t2 = tio.RescaleIntensity(out_min_max=(0, 1))(tio.ScalarImage(t2_path))
            transforms = [
                tio.transforms.RandomMotion(degrees=20, translation=20, num_transforms=4, image_interpolation='linear'),
                tio.Resample(
                    (
                        t2.spacing[0] * down_factor,
                        t2.spacing[1] * down_factor,
                        t2.spacing[2] * down_factor,
                    ),
                    image_interpolation="gaussian",
            ),
            #     tio.Resample( #forces resize to patch size
            #         (
            #             self.patch_size[0],
            #             self.patch_size[0],
            #             t2.spacing[2] * down_factor,
            #         ),
            #         image_interpolation="gaussian",
            # ),
            ]
            t2_lowres = tio.Resample(
                    (
                        t2.spacing[0] * down_factor,
                        t2.spacing[1] * down_factor,
                        t2.spacing[2] * down_factor,
                    ),
                    image_interpolation="gaussian",
            )(t2)
            t2_degrad = tio.transforms.Compose(transforms)(t2)
            subjects.append(tio.Subject(t2=t2_lowres, t2_degrad=t2_degrad))

        #test data (use with care)
        all_baboons = glob.glob(self.baboon_path + '*.nii.gz', recursive=True)[:self.max_subjects]
        baboons = []
        for path in all_baboons:
            baboon_image = tio.RescaleIntensity(out_min_max=(0, 1))(tio.ScalarImage(path))
            #TODO: upscale at this level
            baboons.append(tio.Subject(t2_degrad=baboon_image))
        #split
        subjects_dataset = tio.SubjectsDataset(subjects)
        baboons_dataset = tio.SubjectsDataset(baboons)
        self.subjects_dataset = subjects_dataset

        if self.patch_size is None:
            self.train_ds, self.val_ds = torch.utils.data.random_split(dataset=subjects_dataset, lengths=[self.train_length, self.val_length])
            self.test_ds = baboons_dataset
        else:
            queue_length = 300
            samples_per_volume = 10
            sampler = tio.data.UniformSampler(patch_size=self.patch_size)
            patches_queue = tio.Queue(
                subjects_dataset=subjects_dataset,
                max_length=queue_length,
                samples_per_volume=samples_per_volume,
                sampler=sampler,
                num_workers=multiprocessing.cpu_count()
            )
            self.train_ds, self.val_ds = torch.utils.data.random_split(dataset=patches_queue, lengths=[self.train_length * samples_per_volume, self.val_length * samples_per_volume])
            self.test_ds = baboons_dataset #TODO: patches on baboon ?

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count() if not self.patch_size else 0)

    def val_dataloader(self)-> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=multiprocessing.cpu_count() if not self.patch_size else 0)

    def test_dataloader(self)-> DataLoader:
        return DataLoader(self.test_ds, num_workers=multiprocessing.cpu_count() if not self.patch_size else 0)

    def get_dataset(self, datatype: str = 'train')-> tio.SubjectsDataset:
        if datatype == 'train':
            return self.subjects_dataset
        if datatype == 'val':
            return self.subjects_dataset
        if datatype == 'test':
            return self.test_ds

# a dataset
mri_dataloader = MriDataModule(
    mri_path=config.dhcp_path,
    baboon_path=config.baboon_path,
    batch_size=config.batch_size,
    max_subjects=config.max_subjects,
)
mri_dataloader.setup()

batch = next(iter(mri_dataloader.train_dataloader()))

# model = AutoEncoder(input_sample=batch, num_channels=[4, 8, 16, 32, 32, 16, 8, 4])
model = config.model_class(input_sample=batch, num_channels=config.num_channels)

trainer = pl.Trainer(gpus=gpu, max_epochs=config.max_epochs)

trainer.fit(model, train_dataloaders=mri_dataloader.train_dataloader(), val_dataloaders=mri_dataloader.val_dataloader())

# output 1 prediction, try to get it in test loader at a point

baboon_image = nib.load(
    "data/baboon_seg/sub-Formule_ses-03_T2_HASTE_AX2_12_crop_seg.nii.gz"
)
baboon_image = torch.FloatTensor(baboon_image.get_fdata()).squeeze(-1)
if isinstance(model, UnetConcatenated):
    baboon_image = torch.FloatTensor(np.vstack([baboon_image[:,:96,:], np.zeros((2, 96, 27))]))
# pred = model(baboon_image[:, :, 13].unsqueeze(0).unsqueeze(0).unsqueeze(-1))
pred = model.forward(baboon_image[:, :, 13].unsqueeze(0).unsqueeze(0).squeeze(-1)) #TODO: correct the ugly code
image_pred = pred.detach().numpy().squeeze()
# pred_nii_image = nib.Nifti1Image(image_pred, affine=np.eye(4))
image_list = [baboon_image[:, :, 13], image_pred]
fig, axes = plt.subplots(1, len(image_list))
for i, img in enumerate(image_list):
    axes[i].imshow(img, cmap="gray", origin="lower")
    
plt.savefig('results' + '/' + str(model.logger.version) + '/' + 'baboon_pred.png')
plt.clf()

#output for classic prediction
batch = next(iter(mri_dataloader.get_dataset()))
x = batch['t2_degrad']['data']
y = batch['t2']['data']
if isinstance(model, UnetAdded) or isinstance(model, UnetConcatenated):
    x = x[:, :96, :96, :]
    y = y[:, :96, :96, :]
pred = model.forward(x[:,:,:,34].unsqueeze(0))
#3D case
# x = x[:,:,:,int(x.shape[3] / 2)].detach().numpy().squeeze(0)
# y = y[:,:,:,int(y.shape[3] / 2)].detach().numpy().squeeze(0)
# pred = pred[:,:,:,:,int(pred.shape[4] / 2)].detach().numpy().squeeze(0).squeeze(0)
#2D case
x = x[:,:,:,34].detach().numpy().squeeze(0)
y = y[:,:,:,34].detach().numpy().squeeze(0)
pred = pred.detach().numpy().squeeze(0).squeeze(0)

image_list = [x, y, pred]
fig, axes = plt.subplots(1, len(image_list))
for i, img in enumerate(image_list):
    axes[i].imshow(img, cmap="gray", origin="lower")

plt.savefig('results' + '/' + str(model.logger.version) + '/' + 'x_y_pred.png')
plt.clf()

file_path = 'results' + '/' + str(model.logger.version) + '/'

config.export_to_txt(file_path=file_path)


# nib.save(img=pred_nii_image, filename="predicition.nii.gz")

'''
fig = plt.imshow(image_pred, cmap='gray')
plt.savefig('pred.png')
'''