'''
Current problem, lack of loss convergence:
-Input correctly normalized
-input correctly display image when reshaped
-Tried with 2048 parameters
-leanring rate ?
-test with linked pixels aka same as with the 2D image
-chagne normalization to 0-1 ? check zeronenorm
-Load image directly in tensorboard
-try square image -> nope
-2D image works ? No
-Confronted input to the notebook -> same shape
-Try outermost linear ? No, but Identity works. LazyLinear gives so so results
Learned, pure pytorch MUCH faster, the 2D version did not renew the batch, probably problematic
-only x, Y -> same floue
-Test with cameraman dataset and original loader
-Adapt w0_inital x10 -> yes. For 2D w0_inital = ~500
-passing pints in batches instead of directly works.
-Needs to retest the trainer in 2D, with val
-retest 2D with new batch each time (tos ee if the principle works) -> yes, separating batch in two and renewing works
TODO: adaptable learning rate
'''

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import nn
from siren_pytorch import SirenNet, Siren
import pytorch_lightning as pl
import nibabel as nib
import numpy as np
import skimage
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import multiprocessing
from typing import Tuple, Union, Dict
from dataclasses import dataclass
from PIL import ImageOps, Image

gpu = [0] if torch.cuda.is_available() else []

#config overides default values
@dataclass
class Config:
    #Experiment parameters
    batch_size: int = 84100 #300000
    epochs:int = 100
    image_path: str = 'data/DHCP_seg/sub-CC00060XX03_ses-12501_t2_seg.nii.gz'
    output_size: int = 290
    mgrid_dim:int = 2
    num_workers: int = multiprocessing.cpu_count() #here to avoid segfault when Chloe works
    gradients_acculumation: bool = False
    n_acc_batch: int = 10
    # num_workers = 20

    #SirenModule parameters
    dim_in: int = 2
    dim_hidden: int = 256
    dim_out: int = 1
    num_layers: int = 5
    final_activation: torch.nn.modules.activation = None #torch.nn.Linear(in_features=1, out_features=dim_out)  #TRY: Final activation LazyLinear(out_features = dim_out)
    w0_initial: float = 300.
    learning_rate: float = 0.001

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

#One siren layer
neuron = Siren(
    dim_in = 3,
    dim_out = 256
)

#original pytorch model TODO: conversion to lightning module ?
# model = SirenNet(
#     dim_in = 3,                        # input dimension, ex. 3d coor
#     dim_hidden = 256,                  # hidden dimension
#     dim_out = 1,                       # output dimension, ex. intensity value
#     num_layers = 5,                    # number of layers
#     final_activation = nn.Sigmoid(),   # activation of final layer (nn.Identity() for direct output)
#     w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
# )


class SirenModule(pl.LightningModule):
    '''
    Lightning module from SirenNet
    '''
    def __init__(
        self,
        dim_in: int = 3,                                                    # input dimension, ex. 3d coor
        dim_hidden: int = 256,                                              # hidden dimension
        dim_out: int = 1,                                                   # output dimension, ex. intensity value
        num_layers: int = 5,                                                # number of layers
        final_activation: torch.nn.modules.activation = None, # activation of final layer (nn.Identity() for direct output)
        w0_initial: float = 30.,                                            # different signals may require different omega_0 in the first layer - this is a hyperparameter
        learning_rate: float = 0.001,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.learning_rate = learning_rate
        self.logging = True

        self.model = SirenNet(
            dim_in=dim_in,                        
            dim_hidden=dim_hidden,                  
            dim_out=dim_out,                       
            num_layers=num_layers,                   
            final_activation=final_activation,   
            w0_initial=w0_initial,                   
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def loss(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(y_pred, y)

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx) -> float:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.train_losses.append(loss.detach().cpu().numpy())
        if self.logging:
            self.log("train loss: ", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> float:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.val_losses.append(loss.detach().cpu().numpy())
        if self.logging:
            self.log("val loss: ", loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        pass

class Mri3DImage(Dataset):
    def __init__(self, image_path):
        super().__init__()
        image = nib.load(image_path)
        mgrid = get_mgrid(np.max(image.shape))
        mgrid = mgrid[:,:,:image.shape[2],:]  #Break -1 1 asumption in reshape directions TODO to be converted full numpy code ? make more flexible
        mgrid = get_mgrid(image.shape[0], dim=config.mgrid_dim) #[:,:,150]

        #make tensor
        pixels = torch.FloatTensor(image.get_fdata()[:,:,150])   #[:,:,150]
        pixels = pixels.flatten()
        pixels = (pixels - torch.min(pixels)) / (torch.max(pixels) - torch.min(pixels)) * 2 - 1
        coords = torch.FloatTensor(mgrid)
        coords = coords.reshape(len(pixels), config.mgrid_dim)
        assert len(coords) == len(pixels)
        self.coords = coords
        self.pixels = pixels.unsqueeze(-1)

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):  
        # if idx > 0: raise IndexError

        return self.coords[idx], self.pixels[idx]
        # return self.coords, self.pixels
        
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
        dataset = Mri3DImage(image_path=image_path)

        #hardcoded split
        # self.train_ds, self.val_ds = torch.utils.data.random_split(dataset=dataset, lengths=[17000000, 72300]) 
        # self.train_ds, self.val_ds = torch.utils.data.random_split(dataset=dataset, lengths=[80000, 4100]) 
        
        self.train_ds = dataset
        self.val_ds = dataset

    #TODO: shuffle dataset, sampler
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=config.num_workers, shuffle=True)

    def val_dataloader(self)-> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=config.num_workers)

    def test_dataloader(self)-> DataLoader:
        return DataLoader(self.test_ds, num_workers=config.num_workers)

    def get_dataset(self, datatype: str = 'train'):
        if datatype == 'train':
            return self.train_ds
        if datatype == 'val':
            return self.val_ds
        if datatype == 'test':
            return self.test_ds

config = Config()

datamodule = MriDataModule(image_path=config.image_path, batch_size=config.batch_size)

dataloader = datamodule.train_dataloader()

# def get_cameraman_tensor(sidelength):
#     img = Image.fromarray(skimage.data.camera())        
#     transform = Compose([
#         Resize(sidelength),
#         ToTensor(),
#         Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
#     ])
#     img = transform(img)
#     return img

# def get_mgrid(sidelen, dim=2):
#     '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
#     sidelen: int
#     dim: int'''
#     tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
#     mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
#     mgrid = mgrid.reshape(-1, dim)
#     return mgrid

# class ImageFitting(Dataset):
#     def __init__(self, sidelength):
#         super().__init__()
#         img = get_cameraman_tensor(sidelength)
#         # img = get_foetal_brain_tensor(sidelength)
#         # img = get_astronaut_tensor(sidelength)
#         self.pixels = img.permute(1, 2, 0).view(-1, 1)
#         self.coords = get_mgrid(sidelength, 2)

#     def __len__(self):
#         return 1

#     def __getitem__(self, idx):    
#         if idx > 0: raise IndexError
            
#         return self.coords, self.pixels

# cameraman = ImageFitting(256)
# dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

model = SirenModule(
        dim_in=config.dim_in,
        dim_hidden=config.dim_hidden,
        dim_out=config.dim_out,
        num_layers=config.num_layers,
        final_activation=config.final_activation,
        w0_initial=config.w0_initial,
        learning_rate=config.learning_rate,
) 

# model = SirenNet(
#         dim_in=config.dim_in,
#         dim_hidden=config.dim_hidden,
#         dim_out=config.dim_out,
#         num_layers=config.num_layers,
#         final_activation=config.final_activation,
#         w0_initial=config.w0_initial,
# ) 

trainer = pl.Trainer(gpus=gpu, max_epochs=config.epochs)

trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())

# #Manual loop
# model.cuda()
# optim = torch.optim.Adam(lr=config.learning_rate, params=model.parameters())
# model_input, ground_truth = next(iter(dataloader))
# model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

# if config.gradients_acculumation is False:
#     for epoch in range(config.epochs):
#         # model_output, coords = model(model_input)
#         # model_input, ground_truth = next(iter(dataloader))
#         # model_input, ground_truth = model_input.cuda(), ground_truth.cuda()    
#         model_output = model(model_input)  
#         loss = F.mse_loss(model_output, ground_truth)

#         print("Step %d, Total loss %0.6f" % (epoch, loss))

#         optim.zero_grad()
#         loss.backward()
#         optim.step()

# if config.gradients_acculumation is True:
#     for epoch in range(config.epochs):
#         # model_output, coords = model(model_input)
#         model_input, ground_truth = next(iter(dataloader))
#         model_input, ground_truth = model_input.cuda(), ground_truth.cuda()    
#         model_output = model(model_input)  
#         loss_batch = F.mse_loss(model_output, ground_truth)
#         loss += loss_batch / config.n_acc_batch

#         if epoch % config.n_acc_batch == 0:
#             print("Step %d, Total loss %0.6f" % (epoch, loss))

#             optim.zero_grad()
#             loss.backward()
#             optim.step()

#             loss = F.mse_loss(ground_truth, ground_truth) #set loss back to zero


# model.to('cpu')

#create the validation and output
mgrid = get_mgrid(config.output_size, dim=config.mgrid_dim)

#select 2D plan, reshape to tensor
# mgrid = torch.FloatTensor(mgrid[:,:,int(mgrid.shape[-1] / 2)]).reshape(config.output_size * config.output_size , 3).unsqueeze(0)
# mgrid = torch.FloatTensor(mgrid[:,:,150]).reshape(config.output_size * config.output_size , config.mgrid_dim).unsqueeze(0)
mgrid = torch.FloatTensor(mgrid).reshape(config.output_size * config.output_size , config.mgrid_dim).unsqueeze(0)

pred = model(mgrid)
#reshape to image, take a 2D view
pred = pred.squeeze(0)
pred = pred.reshape(config.output_size, config.output_size)

fig = plt.imshow(pred.detach().numpy(), cmap='gray')
plt.savefig('sirenresult.png')

version_number = str(model.logger.version)

#test saves
model = SirenModule(
        dim_in=config.dim_in,
        dim_hidden=config.dim_hidden,
        dim_out=config.dim_out,
        num_layers=config.num_layers,
        final_activation=config.final_activation,
        w0_initial=config.w0_initial,
        learning_rate=config.learning_rate,
)
model.load_from_checkpoint(f'lightning_logs/version_{version_number}/checkpoints/epoch=0-step=8409.ckpt')

trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())
# torch.save(model.state_dict(), PATH)
# model.load_state_dict(torch.load(PATH))