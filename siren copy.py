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

'''
You will have a problem for final prediction, if you are only able to predict 1 pixel. You need to thinkg carefully about it

The idea right now would be to have a dataloader only returning 1 pixel, but a batch size very large. Makes it adaptable for every hardware/image. Difficulty will be then with prediction (you have to manuallycreate a grid and fill it)

'''

gpu = [0] if torch.cuda.is_available() else []

#One siren layer
neuron = Siren(
    dim_in = 3,
    dim_out = 256
)

#original pytorch model TODO: conversion to lightning module ?
model = SirenNet(
    dim_in = 3,                        # input dimension, ex. 3d coor
    dim_hidden = 256,                  # hidden dimension
    dim_out = 1,                       # output dimension, ex. intensity value
    num_layers = 4,                    # number of layers
    final_activation = nn.Sigmoid(),   # activation of final layer (nn.Identity() for direct output)
    w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
)

def get_mgrid(sidelen, dim=3):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_DHCP_brain_tensor():
    '''
    Cube for simplicity
    '''
    image = nib.load('data/DHCP_seg/sub-CC00060XX03_ses-12501_t2_seg.nii.gz')
    image_np = image.get_fdata()
    shape = image_np.shape
    image_np = np.vstack((image_np.T, np.zeros((shape[0], shape[1], shape[1] - shape[2])).T))  #first dimension of array only is allowed not to be equal = .T
    image_np = image_np[:64, :64, :64] #Taillage à la serpette pour faire passer ça dans la carte
    transform = Compose([
        ToTensor(),
        #Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
    ])
    image = transform(image_np)
    image = (image - torch.min(image))/(torch.max(image)-torch.min(image)) * 2 -1
    return image

class ImageFitting3D(Dataset):
    def __init__(self, sidelength=64):
        super().__init__()
        img = get_DHCP_brain_tensor()
        self.pixels = img.reshape(-1)
        self.coords = get_mgrid(sidelength, dim=3)

    def __len__(self):
        return 1

    def __getitem__(self, idx):     
        if idx > 0: raise IndexError

        return self.coords, self.pixels

image = ImageFitting3D()

dataloader = DataLoader(image, batch_size=10000000, pin_memory=True, num_workers=12)

#Training loop
model.to(0)
total_steps = 500 # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 10

optim = torch.optim.Adam(lr=1e-3, params=model.parameters())

model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
losses = []

for step in range(total_steps):
    model_output = model(model_input)    
    # loss = F.mse_loss(model_output, ground_truth)
    loss = ((model_output - ground_truth)**2).mean()
    losses.append(loss.detach().cpu().numpy())
    
    if not step % steps_til_summary:
        print("Step %d, Total loss %0.6f" % (step, loss))

        # fig, axes = plt.subplots(1,3, figsize=(18,6))
        # axes[0].imshow(model_output.cpu().view(256,256).detach().numpy())
        # plt.show()

    optim.zero_grad()
    loss.backward()
    optim.step()

#create a result output vs gt
ground_truth = ground_truth.detach().cpu().numpy()[:,:,32]
model_output = model(model_input) 
model_output = model_output.detach().cpu().numpy()[:,:,32]

fig, axes = plt.subplots(1,2, figsize=(18,6))
axes[0].imshow(model_output.view(290,290))
axes[1].imshow(ground_truth.view(290,290))
fig.save('sirenresult.png')

