'''
You are working on the 3D version of hte code (2D works both with acc gradient and without)
Also, correct epohcs term: 1 epoch = dataset seen once in its totality !
'''

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import nn
from siren_pytorch import SirenNet, Siren
import nibabel as nib
import numpy as np
import skimage
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import multiprocessing
from typing import Tuple, Union, Dict
from dataclasses import dataclass
from PIL import ImageOps, Image
from skimage import metrics 
import os
import time

#config overides default values
@dataclass
class Config:
    #Experiment parameters
    batch_size: int = 1024 #524288 #300000 max at 512 for 3D, 84100 complete image for 2D
    epochs:int = 100
    image_path: str = 'data/DHCP_seg/sub-CC00060XX03_ses-12501_t2_seg.nii.gz'
    output_size: int = 128
    num_workers: int = multiprocessing.cpu_count() 
    gradients_accumulation: bool = False
    n_acc_batch: int = 10
    renew_batch: bool = True if gradients_accumulation else False #Set to True if the batch does not include the whole image. Set to False for 10x performance

    #SirenModule parameters
    dim_in: int = 2
    dim_hidden: int = 512
    dim_out: int = 1
    num_layers: int = 5
    final_activation: torch.nn.modules.activation = None
    w0_initial: float = 30.
    w0: int = 30.
    learning_rate: float = 0.001

    #output
    output_path:str = 'results_siren/'
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path)
    experiment_number:int = 0 if len(os.listdir(output_path)) == 0 else len(os.listdir(output_path))

    def export_to_txt(self, file_path: str = '') -> None:
        with open(file_path + 'config.txt', 'w') as f:
            for key in self.__dict__:
                f.write(str(key) + ' : ' + str(self.__dict__[key]) + '\n')

def get_mgrid(sidelen, dim=3):
    '''Generates a grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    return mgrid

def get_mgrid(shape):
    tensors = []
    for dim in shape:
        tensor = torch.linspace(-1, 1, steps=dim) * len(shape)
        tensors.append(tensor) 
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    return mgrid


class MriImage(Dataset):
    def __init__(self, image_path):
        super().__init__()
        image = nib.load(image_path)
        image = image.get_fdata()    #[64:192, 64:192, 100:164]
        if config.dim_in == 3:
            x = torch.linspace(-1, 1, steps=image.shape[0])
            y = torch.linspace(-1, 1, steps=image.shape[1])
            z = torch.linspace(-1, 1, steps=image.shape[2])
            mgrid = torch.stack(torch.meshgrid(x,y,z), dim=-1)
        if config.dim_in == 2:
            x = torch.linspace(-1, 1, steps=image.shape[0])
            y = torch.linspace(-1, 1, steps=image.shape[1])
            mgrid = torch.stack(torch.meshgrid(x,y,z), dim=-1)

        #create data tensors
        pixels = torch.FloatTensor(image)
        if config.dim_in == 2:
            pixels = pixels[:,:,int(pixels.shape[2] / 2)]
        pixels = pixels.flatten()
        pixels = (pixels - torch.min(pixels)) / (torch.max(pixels) - torch.min(pixels)) * 2 - 1
        coords = torch.FloatTensor(mgrid)
        coords = coords.reshape(len(pixels), config.dim_in)
        assert len(coords) == len(pixels)
        self.coords = coords
        self.pixels = pixels.unsqueeze(-1)

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):  
        return self.coords[idx], self.pixels[idx]

class MriDataModule(pl.LightningDataModule):
    '''
    Take ONE mri image and returns coords and pixels
    '''
    def __init__(
        self,
        image_path: str,
        batch_size: int,
        *args,
        **kwargs
    ):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.image_path = image_path
        self.batch_size = batch_size
        dataset = MriImage(image_path=image_path)

        #hardcoded split
        # self.train_ds, self.val_ds = torch.utils.data.random_split(dataset=dataset, lengths=[17000000, 72300]) 
        # self.train_ds, self.val_ds = torch.utils.data.random_split(dataset=dataset, lengths=[80000, 4100]) 
        
        self.train_ds = dataset
        self.val_ds = dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=config.num_workers, shuffle=True)

    def val_dataloader(self)-> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=config.num_workers)

    def test_dataloader(self)-> DataLoader:
        return DataLoader(self.test_ds, num_workers=config.num_workers)

config = Config()

datamodule = MriDataModule(image_path=config.image_path, batch_size=config.batch_size)

dataloader = datamodule.train_dataloader()

model = SirenNet(
        dim_in=config.dim_in,
        dim_hidden=config.dim_hidden,
        dim_out=config.dim_out,
        num_layers=config.num_layers,
        final_activation=config.final_activation,
        w0 = config.w0,
        w0_initial=config.w0_initial,
) 

###############
#TRAINING LOOP#
###############
losses = []
model.cuda()
optim = torch.optim.Adam(lr=config.learning_rate, params=model.parameters())
# loss = F.mse_loss(ground_truth, ground_truth)
training_start = int(time.time())
if config.gradients_accumulation is False:
    for epoch in range(config.epochs):
        for model_input, ground_truth in dataloader:
            model_input, ground_truth = model_input.cuda(), ground_truth.cuda()    
            model_output = model(model_input)  
            loss = F.mse_loss(model_output, ground_truth)
            losses.append(loss.detach().cpu().numpy())

            print("Step %d, Total loss %0.6f" % (epoch, loss))

            optim.zero_grad()
            loss.backward()
            optim.step()

if config.gradients_accumulation is True: #successfully in 2D
    for epoch in range(config.epochs):
        for model_input, ground_truth in dataloader:
            model_input, ground_truth = model_input.cuda(), ground_truth.cuda()    
            model_output = model(model_input)
            loss = F.mse_loss(model_output, ground_truth) / config.n_acc_batch
            loss.backward()
            if epoch % config.n_acc_batch == 0:
                print("Step %d, Total loss %0.6f" % (epoch, loss))
                optim.step()
                optim.zero_grad()
                losses.append(loss.detach().cpu().numpy())

training_stop = int(time.time())
######################
#END OF TRAINING LOOP#
######################

filepath = config.output_path + str(config.experiment_number) + '/'

if os.path.isdir(filepath) is False:
    os.mkdir(filepath)

model.to('cpu')
torch.save(model.state_dict(), filepath + 'checkpoint.pt')

#ground truth
ground_truth = nib.load(config.image_path)
ground_truth = ground_truth.get_fdata(dtype=np.float32) #[:,:,int(ground_truth.shape[-1] / 2)] [64:128, 64:128, 96]
ground_truth = (ground_truth - np.min(ground_truth)) / (np.max(ground_truth) - np.min(ground_truth)) * 2 - 1
#nifti here
if len(ground_truth.shape) == 3:
    ground_truth = ground_truth[:,:,int(ground_truth.shape[-1] / 2)]
fig = plt.imshow(ground_truth, cmap='gray', vmin=-1.0, vmax=1.0)
plt.savefig(filepath + 'ground_truth.png')
plt.clf()

#create the validation and output
if config.dim_in == 3:
    x = torch.linspace(-1, 1, steps=ground_truth.shape[0])
    y = torch.linspace(-1, 1, steps=ground_truth.shape[1])
    z = torch.linspace(-1, 1, steps=ground_truth.shape[2])
    mgrid = torch.stack(torch.meshgrid(x,y,z), dim=-1)
if config.dim_in == 2:
    x = torch.linspace(-1, 1, steps=ground_truth.shape[0])
    y = torch.linspace(-1, 1, steps=ground_truth.shape[1])
    mgrid = torch.stack(torch.meshgrid(x,y,z), dim=-1)
    
mgrid = torch.FloatTensor(mgrid).reshape(ground_truth * ground_truth , config.dim_in).unsqueeze(0)

pred = model(mgrid)
#reshape to image, take a 2D view
pred = pred.squeeze(0)
pred = pred.reshape(ground_truth.shape)
pred = pred.detach().numpy()
#nifti here
if len(pred.shape) == 3:
    pred = pred[:,:,int(pred.shape[-1] / 2)]
fig = plt.imshow(pred, cmap='gray', vmin=-1.0, vmax=1.0)
plt.savefig(filepath + 'results_sample.png')
plt.clf()

#Difference image
diff = ground_truth - pred
fig = plt.imshow(diff, cmap='gray', vmin=-1.0, vmax=1.0)
plt.savefig(filepath + 'difference_image.png')
plt.clf()

lossfig = plt.plot(range(len(losses)), losses, color='r')
plt.savefig(filepath + 'losses.png')

config.export_to_txt(filepath)

#metrics
with open (filepath + 'scores.txt', 'w') as f:
    f.write('MSE : ' + str(metrics.mean_squared_error(ground_truth, pred)) + '\n')
    f.write('PSNR : ' + str(metrics.peak_signal_noise_ratio(ground_truth, pred)) + '\n')
    f.write('SSMI : ' + str(metrics.structural_similarity(ground_truth, pred)) + '\n')
    f.write('training time  : ' + str(training_stop - training_start) + ' seconds' + '\n')
    f.write('Number of trainable parameters : ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + '\n') #remove condition if you want total parameters
    f.write('Max memory allocated : ' + str(torch.cuda.max_memory_allocated()) + '\n')


