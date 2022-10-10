'''
You are working on the 3D version of hte code (2D works both with acc gradient and without)
Also, correct epohcs term: 1 epoch = dataset seen once in its totality !
'''

import torch
# import pytorch_lightning as pl
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
from einops import rearrange
import math

#config overides default values
@dataclass
class Config:
    #Experiment parameters
    batch_size: int = 300000 #524288 #300000 max at 512 for 3D, 84100 complete image for 2D
    epochs:int = 150
    image_path: str = 'data/DHCP_seg/sub-CC00060XX03_ses-12501_t2_seg.nii.gz'
    num_workers: int = multiprocessing.cpu_count() 
    device = [0] if torch.cuda.is_available() else []

    #SirenModule parameters
    dim_in: int = 3
    dim_hidden: int = 512
    dim_out: int = 1
    num_layers: int = 5
    final_activation: torch.nn.modules.activation = None
    w0_initial: float = 30.
    w0: int = 30.
    learning_rate: float = 0.001

    #output
    output_size: int = 290
    output_path:str = 'results_siren/'
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path)
    experiment_number:int = 0 if len(os.listdir(output_path)) == 0 else len(os.listdir(output_path))

    def export_to_txt(self, file_path: str = '') -> None:
        with open(file_path + 'config.txt', 'w') as f:
            for key in self.__dict__:
                f.write(str(key) + ' : ' + str(self.__dict__[key]) + '\n')

def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# siren layer
class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
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

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


# siren network
class SirenNet(pl.LightningModule):
    def __init__(
        self,
        dim_in=3,
        dim_hidden=128,
        dim_out=1,
        num_layers=2,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.losses = []

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(
                Siren(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                )
            )

        final_activation = (
            nn.Identity() if final_activation is None else final_activation
        )
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
        )

    def forward(self, x, mods=None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if mod is not None:
                x *= rearrange(mod, "d -> () d")

        return self.last_layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)

        loss = F.mse_loss(z, y)
        self.losses.append(loss.detach().cpu().numpy())

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=0)
        return optimizer

class MriImage(Dataset):
    def __init__(self, image_path):
        super().__init__()
        image = nib.load(image_path)
        image = image.get_fdata()   #[64:192, 64:192, 100:164]
        if config.dim_in == 3:
            x = torch.linspace(-1, 1, steps=image.shape[0])
            y = torch.linspace(-1, 1, steps=image.shape[1])
            z = torch.linspace(-1, 1, steps=image.shape[2])
            mgrid = torch.stack(torch.meshgrid(x,y,z), dim=-1)
        if config.dim_in == 2:
            x = torch.linspace(-1, 1, steps=image.shape[0])
            y = torch.linspace(-1, 1, steps=image.shape[1])
            mgrid = torch.stack(torch.meshgrid(x,y), dim=-1)

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
        self.test_ds = dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=config.num_workers, shuffle=True)

    def val_dataloader(self)-> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=config.num_workers)

    def test_dataloader(self)-> DataLoader:
        return DataLoader(self.test_ds, num_workers=config.num_workers)

config = Config()

datamodule = MriDataModule(image_path=config.image_path, batch_size=config.batch_size)

train_loader = datamodule.train_dataloader()

model = SirenNet(
        dim_in=config.dim_in,
        dim_hidden=config.dim_hidden,
        dim_out=config.dim_out,
        num_layers=config.num_layers,
        final_activation=config.final_activation,
        w0 = config.w0,
        w0_initial=config.w0_initial,
) 

trainer = pl.Trainer(gpus=config.device, max_epochs=config.epochs)
training_start = int(time.time())
trainer.fit(model, train_loader)
training_stop = int(time.time())

filepath = config.output_path + str(config.experiment_number) + '/'

if os.path.isdir(filepath) is False:
    os.mkdir(filepath)

trainer.save_checkpoint(filepath + 'checkpoint.ckpt')

#ground truth
ground_truth_image = nib.load(config.image_path)
ground_truth = ground_truth_image.get_fdata(dtype=np.float32) #[:,:,int(ground_truth.shape[-1] / 2)] [64:128, 64:128, 96]
ground_truth = (ground_truth - np.min(ground_truth)) / (np.max(ground_truth) - np.min(ground_truth)) * 2 - 1
#nifti here
if len(ground_truth.shape) == 3:
    ground_truth = ground_truth[:,:,int(ground_truth.shape[-1] / 2)]
fig = plt.imshow(ground_truth, cmap='gray', vmin=-1.0, vmax=1.0)
plt.savefig(filepath + 'ground_truth.png')
plt.clf()

#create the validation and output
if config.dim_in == 3:
    x = torch.linspace(-1, 1, steps=ground_truth_image.shape[0])
    y = torch.linspace(-1, 1, steps=ground_truth_image.shape[1])
    z = torch.linspace(-1, 1, steps=ground_truth_image.shape[2])
    mgrid = torch.stack(torch.meshgrid(x,y,z), dim=-1)
    mgrid = torch.FloatTensor(mgrid).reshape(ground_truth_image.shape[0] * ground_truth_image.shape[1] * ground_truth_image.shape[2] , config.dim_in)

if config.dim_in == 2:
    x = torch.linspace(-1, 1, steps=ground_truth_image.shape[0])
    y = torch.linspace(-1, 1, steps=ground_truth_image.shape[1])
    mgrid = torch.stack(torch.meshgrid(x,y), dim=-1)
    mgrid = torch.FloatTensor(mgrid).reshape(ground_truth_image.shape[0] * ground_truth_image.shape[1] , config.dim_in)

# pred = torch.concat(trainer.predict(model, mgrid))
test_dataset = torch.utils.data.TensorDataset(mgrid, mgrid)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

pred = torch.concat(trainer.predict(model, test_loader))
#reshape to image, take a 2D view
if config.dim_in == 3:
    pred = pred.reshape(ground_truth_image.shape)
if config.dim_in == 2:
    pred =pred.reshape(ground_truth.shape)
pred = pred.detach().numpy()
#nifti here
nib.save(nib.Nifti1Image(pred, ground_truth_image.affine), filepath + 'prediction.nii.gz') 
if config.dim_in == 3:
    pred = pred[:,:,int(pred.shape[-1] / 2)]
fig = plt.imshow(pred, cmap='gray', vmin=-1.0, vmax=1.0)
plt.savefig(filepath + 'results_sample.png')
plt.clf()

#create an upsampling
if config.dim_in == 3:
    x = torch.linspace(-1, 1, steps=config.output_size)
    y = torch.linspace(-1, 1, steps=config.output_size)
    z = torch.linspace(-1, 1, steps=config.output_size)
    mgrid = torch.stack(torch.meshgrid(x,y,z), dim=-1)
    mgrid = torch.FloatTensor(mgrid).reshape(config.output_size * config.output_size* config.output_size , config.dim_in)

if config.dim_in == 2:
    x = torch.linspace(-1, 1, steps=config.output_size)
    y = torch.linspace(-1, 1, steps=config.output_size)
    mgrid = torch.stack(torch.meshgrid(x,y), dim=-1)
    mgrid = torch.FloatTensor(mgrid).reshape(config.output_size * config.output_size , config.dim_in)

up_dataset = torch.utils.data.TensorDataset(mgrid, mgrid)
up_loader = torch.utils.data.DataLoader(up_dataset, batch_size=config.batch_size, num_workers=config.num_workers)

upsampling = torch.concat(trainer.predict(model, up_loader))
#reshape to image, take a 2D view
if config.dim_in == 2:
    upsampling = upsampling.reshape(config.output_size, config.output_size)
if config.dim_in == 3:
    upsampling = upsampling.reshape(config.output_size, config.output_size, config.output_size)
upsampling = upsampling.detach().numpy()
#nifti here
nib.save(nib.Nifti1Image(upsampling, ground_truth_image.affine), filepath + 'upsampling.nii.gz') 
if config.dim_in == 3:
    upsampling = upsampling[:,:,int(pred.shape[-1] / 2)]
fig = plt.imshow(upsampling, cmap='gray', vmin=-1.0, vmax=1.0)
plt.savefig(filepath + 'upsampling.png')
plt.clf()

#Difference image
diff = ground_truth - pred
fig = plt.imshow(diff, cmap='gray', vmin=-1.0, vmax=1.0)
plt.savefig(filepath + 'difference_image.png')
plt.clf()

lossfig = plt.plot(range(len(model.losses)), model.losses, color='r')
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


