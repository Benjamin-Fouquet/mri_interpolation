'''
Data preparation and dataloader
'''

from dataclasses import dataclass
import torchio as tio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import multiprocessing
import pytorch_lightning as pl
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import argparse
import torchvision
import glob
import sys
from skimage import metrics #mean_squared_error, peak_signal_noise_ratio
import os

@dataclass
class Data():
    '''
    Placeholder for datacalss in case you need it one day
    '''
    epochs: int
    percentage: int
    pass

#parameters for queue
queue_length = 300
samples_per_volume = 10
n_max_subjects = 100

subjects = []


all_t2s = glob.glob(input_path + '*_T2.nii.gz', recursive=True)[:n_max_subjects]
all_masks = glob.glob(input_path + '*_mask.nii.gz', recursive=True)[:n_max_subjects]


if all_t2s == [] or all_masks == []:
    print('Failed to import data, verify datapath')
    sys.exit()

#loop for subjects
for mask in all_masks:
    id_subject = mask.split('/')[2].split('_')[0:2][0]  #ugly hard coded split, TODO: REGEX ?

    t2_file = [s for s in all_t2s if id_subject in s][0]
    
    subject = tio.Subject(
        t2=tio.ScalarImage(t2_file),
        seg=tio.LabelMap(mask),
    )
    subjects.append(subject)

transforms = [

    tio.RescaleIntensity(out_min_max=(0, 1)),
    #tio.transforms.ZNormalization(masking_method='label'),
    #tio.RandomAffine()
]

#mask created before normalisation
def create_rn_mask(subject, percentage):
    shape = subject.t2.shape
    rn_mask = torch.FloatTensor(
        np.random.choice(
            [1, 0], size=shape, p=[(percentage * 0.01), 1 - ((percentage * 0.01))]
        )
    )
    undersampling = rn_mask * subject.t2.data
    subject.add_image(tio.LabelMap(tensor=rn_mask, affine=subject.t2.affine), "rn_mask")
    subject.add_image(
        tio.ScalarImage(tensor=undersampling, affine=subject.t2.affine), "rn_t2"
    )
    return None

for subject in subjects:
    create_rn_mask(subject, percentage=percentage)

transform = tio.Compose(transforms)

subjects_dataset = tio.SubjectsDataset(
    subjects, transform=transform
)  # only accept lists

sampler = tio.data.UniformSampler(patch_size)
patches_queue = tio.Queue(
    subjects_dataset=subjects_dataset,
    max_length=queue_length,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=num_workers,
)

patches_loader = DataLoader(
    dataset=patches_queue,
    batch_size=batch_size,
    num_workers=0, #must be 0
)
###
#Grid sampler for sampling accross 1 unique volume
###
# grid_sampler = tio.inference.GridSampler(
#     subject,
#     patch_size,
#     patch_overlap
#     )

# patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size, num_workers=num_workers)
# aggregator = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

dataloader = DataLoader(subjects_dataset, num_workers=num_workers, batch_size=1) #used for test


