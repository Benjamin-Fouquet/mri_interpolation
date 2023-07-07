import vtk
import argparse
from scipy.interpolate import griddata, interpn
from torch import autograd as ag
import torch
import glob
import nibabel as nib
from scipy.fft import fftn, ifftn, fftshift, fftfreq

def nii_2_mesh(filename_nii, filename_stl, label):

    """
    Read a nifti file including a binary map of a segmented organ with label id = label. 
    Convert it to a smoothed mesh of type stl.

    filename_nii     : Input nifti binary map 
    filename_stl     : Output mesh name in stl format
    label            : segmented label id 
    """

    # read the file
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filename_nii)
    reader.Update()
    
    # apply marching cube surface generation
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(reader.GetOutputPort())
    surf.SetValue(0, label) # use surf.GenerateValues function if more than one contour is available in the file	
    surf.Update()
    
    #  auto Orient Normals
    surf_cor = vtk.vtkPolyDataNormals()
    surf_cor.SetInputConnection(surf.GetOutputPort())
    surf_cor.ConsistencyOn()
    surf_cor.AutoOrientNormalsOn()
    surf_cor.SplittingOff()
    surf_cor.Update()
    
    #smoothing the mesh
    smoother= vtk.vtkWindowedSincPolyDataFilter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        smoother.SetInput(surf_cor.GetOutput())
    else:
        smoother.SetInputConnection(surf_cor.GetOutputPort())
    smoother.SetNumberOfIterations(60)
    smoother.NonManifoldSmoothingOn()
    #smoother.NormalizeCoordinatesOn() #The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
    smoother.GenerateErrorScalarsOn()
    smoother.Update()
     
    # save the output
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(smoother.GetOutputPort())
    writer.SetFileTypeToASCII()
    writer.SetFileName(filename_stl)
    writer.Write()

#extract (x, y, z), how many points ? 

t_zero = torch.FloatTensor((0, 0, 0))
t_one = torch.FloatTensor((1, 1, 1))
t_zero.requires_grad = True
t_one.requires_grad = True

delta_t = t_one - t_zero

ag.grad(outputs=delta_t, inputs=t_one)

from typing import List, Optional, Union
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
import tinycudann as tcnn 
import torch
import pytorch_lightning as pl 
import torch.nn.functional as F
import json
import nibabel as nib 
from dataclasses import dataclass
import os
from types import MappingProxyType
import numpy as np
import math
import rff
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.nn
from functools import lru_cache
import torch.utils.data
import matplotlib.pyplot as plt

torch.manual_seed(1337)

@dataclass
class BaseConfig:
    checkpoint_path: Optional[str] = None #'lightning_logs/version_269/checkpoints/epoch=285-step=16874.ckpt' #'lightning_logs/version_384/checkpoints/epoch=99-step=100.ckpt'
    # image_path: str = '/mnt/Data/FetalAtlas/template_T2.nii.gz'
    image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_shape = nib.load(image_path).shape
    batch_size: int = 50000 #~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 15
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None 
    # Network parameters
    encoder_type: str = 'Siren' #   
    n_frequencies: int = 512  #for classic, n_out = 2 * n_freq. For tcnn, n_out = 2 * n_freq * dim_in
    n_frequencies_t: int = 8 if encoder_type == 'tcnn' else 16
    dim_in: int = len(image_shape)
    dim_hidden: int = 256 
    dim_out: int = 1
    num_layers: int = 8
    skip_connections: tuple = () #(5, 11,)
    lr: float = 1e-3  # G requires training with a custom lr, usually lr * 0.1 
    interp_factor: int = 2

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch size", type=int, required=False)
    parser.add_argument("--epochs", help="Number of epochs", type=int, required=False)
    parser.add_argument("--image_path", help="path of image", type=str, required=False)
    parser.add_argument("--encoder_type", help="tcnn or classic", type=str, required=False)
    parser.add_argument("--dim_hidden", help="size of hidden layers", type=int, required=False)
    parser.add_argument("--num_layers", help="number of layers", type=int, required=False)
    
    args = parser.parse_args()

def export_to_txt(dict: dict, file_path: str = "") -> None:
    '''
    Helper function to export dictionary to text file
    '''
    with open(file_path + "config.txt", "a+") as f:
        for key in dict:
            f.write(str(key) + " : " + str(dict[key]) + "\n")

config = BaseConfig()

# parsed argument -> config
for key in args.__dict__:
    if args.__dict__[key] is not None:
        config.__dict__[key] = args.__dict__[key]

#utils for siren
def exists(val):
    return val is not None


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Sine(torch.nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(torch.nn.Module):
    '''
    Siren layer
    '''
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=10.0,
        c=10.0,
        is_first=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        phase = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = torch.nn.Parameter(weight)
        self.phase = torch.nn.Parameter(phase)
        self.bias = torch.nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out + self.phase) 
        return out
    
siren = Siren(1, 1)

out = siren(x.unsqueeze(-1))
out = out.detach().numpy()

plt.plot(range(len(x)), [func(x) for x in x])
plt.savefig('out.png')
plt.clf()

#extract i

#Fourier

#Hash enco

def func(x):
    return 1 if x > 0.2 and x < 0.6 else 0


enco1 = GaussianFourierFeatureTransform(2)

enco2 = rff.layers.GaussianEncoding(sigma=10.0, input_size=2, encoded_size=256)

for idx, row in enumerate(im):
    im[idx] = np.sin(idx)

plt.imshow(im)
plt.savefig('out.png')

axes = []
for s in im.shape:
    axes.append(torch.linspace(0, 1, s))
    
mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

path = '/home/benjamin/results_repaper/version_30/'

image = nib.load(path + 'interpolation.nii.gz')
data = image.get_fdata(dtype=np.float32)

# data = data[:,:,3,:]

fig, axes = plt.subplots(6, 5)

for j in range(5):
    for i in range(6):
        slice = data[..., (j * 6) + i]
        axes[i][j].imshow(slice.T, origin="lower", cmap="gray") #cmap="gray"
        # print((j * 6) + i)
        
plt.savefig('out.png')


#extraction of mean for foot

#glob all the names for movieclear in WD
path = '/media/benjamin/WD Elements/Patty/sourcedata/'

subjects = glob.glob(path + 'sub*', recursive=True)

#create a list of all images with the name 'MovieClear' in them. Not prefect but you dont need perfect mean
images_list = []
for subject in subjects:
    images_list.append(glob.glob(subject + '/' + '**_MovieClear_**', recursive=True))
   
#flatten
images_list = [item for sublist in images_list for item in sublist] 
    
#mean all images in 4D, check if correct resolution

mean = np.zeros((352, 352, 6, 15), dtype=np.float32)
for path in images_list:
    try:
        image = nib.load(path)
    except:
        pass
    
    if image.shape == (352, 352, 6, 15):
        mean += image.get_fdata(dtype=np.float32)
        #do something
    else:
        pass

#save
nib.save(nib.Nifti1Image(mean, affine=image.affine), 'mean_foot_dynamic.nii.gz') 

    
im = nib.load('/home/benjamin/Documents/Datasets/sub_E01_dynamic_MovieClear_active_run_12.nii.gz').get_fdata()

fourier = fftshift(fftn(im))

#get amplitude ?
abs = np.sqrt(fourier.real ** 2 + fourier.imag **2)

freq = fftfreq(fourier.size, d=1)

plt.imshow(abs[:,:,3, 7].real)
plt.show()
plt.clf()