'''
REPTILE and MAML classes

Maml has to be class, it holds both data and process, mainly data related to the second optimizer


Reptile could be a simple function, it only holds the mean
Can offer several initialization schemes for mri (mean brain, mean form, mean foot) for instance. Do class as easy. If offering specific init, becomes specific to the model and loader
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

'''
REPTILE:
Optimal init for INR = mean image done prior to training

Two classes for reptile (one fo INR one for image ?). Would probably make sense
'''
class Reptile:
    def __init__(self, model, dataset, initialization_mode, batch_size) -> None:
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        
        #create a loader
        if initialization_mode == 'mean':
            self.loader = self.create_mean(dataset)
            
        elif initialization_mode == 'random':
            self.loader = self.create_random(dataset)
            
        elif initialization_mode == 'mri_foot':
            mean = nib.load('/mnt/Data/Equinus_BIDS_dataset/mean_foot_dynamic.nii.gz').get_fdata(np.float32)
            Y = torch.FloatTensor(mean).reshape(-1, 1)
            Y = Y / Y.max()

            axes = []
            for s in mean.shape:
                axes.append(torch.linspace(0, 1, s))

            mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

            coords = torch.FloatTensor(mgrid)
            X = coords.reshape(len(Y), len(mean.shape))

            dataset = torch.utils.data.TensorDataset(X, Y)
            
            return torch.utils.DataLoader(dataset, batch_size=self.batch_size, suffle=True)
            
        elif initialization_mode == 'mri_brain':
            pass
        
    def create_mean(dataset):
        x, y = next(iter(dataset))
        mean = torch.zeros(y.shape)
        #differential treatment depening on dataset instance style
        return mean #should be loader
        
    def optimize(self, n_steps):
        #make mean of the loader, delegate
        
        #fit the model to the mean
        trainer = pl.Trainer(
        gpus=[0] if torch.cuda.is_available else [],
        max_epochs=n_steps,
        precision=32,
        # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
        )
        
        trainer.fit(self.model, self.loader)
        
        return self.model
    
    # (model, loader)
    #class or function. Do I need data in the function ?
    
    #Think through for it to be arbitratry model and loader, pass dataset instead? Think MNIST for toy example
        #Problem: for implicit representation, mean point has no meaning, each point carries widely varying information, and in the case of H the table is very specific to the point
            #The aim is to learn an optimal initialization, can yu do better than normal init ? Where to find info ?
                #existence of a mean H table for the network to initiliaze upon ?
                    #If so, hwo to build it ?
    
    #make a mean of the loader
        #what mean to do exactly ? What does make sense?
    
    #present the mean to the model

    #fit the model to the mean
    

class MAML:
    Model.encode()
    loader
    optimizer
    #choose a number of steps
    
    #optimize on 5 steps
    
    #calculate last loss
    
    #Smallest version of maml possible ?
    
