from typing import List, Optional, Union
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
# import tinycudann as tcnn 
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
import tinycudann as tcnn
from models import SirenNet

torch.manual_seed(1337)


@dataclass
class BaseConfig:
    checkpoint_path = None #'lightning_logs/version_61/checkpoints/epoch=199-step=12000.ckpt'
    # image_path: str = '/mnt/Data/FetalAtlas/template_T2.nii.gz'
    image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_shape = nib.load(image_path).shape
    interp_shape = (352, 352, 30)
    batch_size: int = 50000 #~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 50
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None 
    # Network parameters
  
    dim_in: int = len(image_shape)
    dim_hidden: int = 256 
    dim_out: int = 1
    num_layers: int = 6
    lr: float = 1e-4  # G requires training with a custom lr, usually lr * 0.1 

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
    parser.add_argument("--n_frequencies", help="number of encoding frequencies", type=int, required=False)
    parser.add_argument("--n_frequencies_t", help="number of encoding frequencies for time", type=int, required=False)
    parser.add_argument("--lr", help="learning rate", type=int, required=False)
    parser.add_argument("--dim_hidden", help="hidden dimension for decoder", type=int, required=False)
    parser.add_argument("--num_layers", help="number of layers for decoder", type=int, required=False)

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
        
          
mri_image = nib.load(config.image_path)

data = mri_image.get_fdata(dtype=np.float32)
data = data[:,:,3,:] #optional line for 2D + t experiments, speedup
config.image_shape = data.shape
config.dim_in = len(data.shape)

#interpolation tests
# data = data[..., ::2]

model = SirenNet(dim_in=config.dim_in, 
                dim_hidden=config.dim_hidden, 
                dim_out=config.dim_out, 
                n_layers=config.num_layers,
                lr=config.lr)

Y = torch.FloatTensor(data).reshape(-1, 1)
Y = Y / Y.max() * 2 - 1

axes = []
for s in config.image_shape:
    axes.append(torch.linspace(-1, 1, s))

mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

coords = torch.FloatTensor(mgrid)

if data.shape[-1] < 15: #conditional step if interp on even frames
    if config.dim_in ==3:
        coords = coords[:,:,::2,:]
    if config.dim_in ==4:
        coords = coords[:,:,:,::2,:]
X = coords.reshape(len(Y), config.dim_in)

dataset = torch.utils.data.TensorDataset(X, Y)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
   
trainer = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
    precision=32,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)
  
trainer = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
    precision=32,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)
trainer.fit(model, train_loader)

filepath = model.logger.log_dir + '/'
config.log = str(model.logger.version)

#create a prediction
pred = torch.concat(trainer.predict(model, test_loader))
            
# im = pred.reshape(config.image_shape)
im = pred.reshape(data.shape)
im = im.detach().cpu().numpy()
im = np.array(im, dtype=np.float32)
if len(im.shape) == 2:
    plt.imshow(im.T)
    plt.savefig(filepath + 'pred.png')
else:
    nib.save(nib.Nifti1Image(im, affine=np.eye(4)), filepath + 'pred.nii.gz')

interp_shapes = [(352, 352, 15), (352, 352, 30), (352, 352, 60)]
# interp_shapes = [(117, 159, 126, 30)]
#ugly loop as placeholder
for shape in interp_shapes:    
    #dense grid
    config.interp_shape = shape
    Y_interp = torch.zeros((np.prod(config.interp_shape), 1))

    axes = []
    for s in config.interp_shape:
        axes.append(torch.linspace(-1, 1, s))
            
    mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

    coords = torch.FloatTensor(mgrid)
    X_interp = coords.reshape(len(Y_interp), config.dim_in)    

    interp_dataset = torch.utils.data.TensorDataset(X_interp, Y_interp)
    interp_loader = torch.utils.data.DataLoader(interp_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    #create an interpolation
    interp = torch.concat(trainer.predict(model, interp_loader))

    interp_im = interp.reshape(config.interp_shape)
        
    interp_im = interp_im.detach().cpu().numpy()
    interp_im = np.array(interp_im, dtype=np.float32)
    nib.save(nib.Nifti1Image(interp_im, affine=np.eye(4)), filepath + f'interpolation{shape}.nii.gz')

config.export_to_txt(file_path=filepath)

output = im
ground_truth = nib.load(config.image_path).get_fdata()
ground_truth = ground_truth / ground_truth.max() * 2 - 1

