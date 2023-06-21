'''
Current status
MLP and Siren ok. Be mindful of normalisation when usign one or the other
Convergence happens
end result coherent but far from needed quality
Check if indices of predicitons match indices of verts for generation of form

TEST ONE: f(idx, t) = (x, y, z)
Test object:
    deforming cube
Tested Networks:
    MLP
    Siren
Conclusion:
    Convergence happens when using RMSE, end result not satisfactory
Potential pitfalls
    Indices of vertices may not be fixed, create issue when reconstructing volume. TODO: how to test this ? Solution, reco extract from dataset (ok). Prob at shuffle on training
    
TEST TWO: f(x, y, z, t) = delta (x, y, z)
Test object:
    deforming cube
Tested Networks:
    MLPDelta
    SirenDelta
Conclusion:
    Works......but of course, because optimal solution is all to zero in this implementation, fails
Pitfalls:
    optimal solution is zeroing all paramters, need reimplementation
    
TODO 26.04:
Reimplement delta calculation to have a true convergence on networks.
'''

import trimesh
import numpy as np
import nibabel as nib
from skimage import measure
from torch.utils.data import TensorDataset, DataLoader
import torch
import os
import pytorch_lightning as pl
from types import MappingProxyType
from dataclasses import dataclass
import torch.nn.functional as F
import torch.nn as nn
import math
import json

@dataclass
class BaseConfig:
    checkpoint_path = None #'lightning_logs/version_384/checkpoints/epoch=99-step=100.ckpt'
    image_path: str = 'data/cube.nii.gz'
    image_shape = nib.load(image_path).shape
    batch_size: int = 10000 #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 50
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None #MappingProxyType({200: 2}) #MappingProxyType({0: 5})

    hashconfig_path: str = 'config/hash_config.json'

    # Network parameters
    dim_in: int = 2
    dim_hidden: int = 128
    dim_out: int = 3
    num_layers: int = 5
    w0: float = 30.0
    w0_initial: float = 30.0
    use_bias: bool = True
    final_activation = None
    lr: float = 1e-3  # G requires training with a custom lr, usually lr * 0.1 
    # datamodule: pl.LightningDataModule = MriFramesDataModule
    # model_cls: pl.LightningModule = MultiHashMLP  
    n_frames: int = 15
    # n_frames: int = image_shape[-1] if len(image_shape) == 4 else None

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")
                
#utils for siren
def exists(val):
    return val is not None


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    '''
    Siren layer
    '''
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
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

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


# siren network
class SirenNet(pl.LightningModule):
    '''
    PURPOSE:
        Implicit representation of arbitrary functions. Mainly used for 2D, 3D image interpolation
    ATTRIBUTES:
        dim_in: dimension of input
        dim_hidden: dimmension of hidden layers. 128-256 are recommended values
        dim_out: dimension of output
        num_layers: number of layers
        w0: multiplying factor so that f(x) = sin(w0 * x) between layers. Recommended value, 30.0
        w0_initial: see w0, recommended value 30.0 as per paper (ref to be found)
        use_bias: if bias is used between layers, usually set to True
        final_activation: flexible last activation for tasks other than interpolation. None means identity
        lr: recommended 1e-4
        layers: layers of the model, minus last layer
        last_layer: final layer of model
        losses: list of losses during training
    METHODS:
        forward: forward pass
        training step: forward pass + backprop
        predict step: used for inference
        configure_optimizer: optimizer linked to model
    '''
    def __init__(
        self,
        dim_in: int = 3,
        dim_hidden: int = 64,
        dim_out: int = 1,
        num_layers: int = 4,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
        lr=1e-4,
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.lr = lr

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
            nn.Identity() if not exists(final_activation) else final_activation
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

            if exists(mod):
                # x *= rearrange(mod, "b d -> b () d") #From Quentin: "d -> () d" -> "b d ->b () d" allors for several batches (images) to be processed
                x *= mod

        return self.last_layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)

        loss = torch.sqrt(F.mse_loss(z, y))

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer

    def set_parameters(self, theta):
        """
        Manually set parameters using matching theta, not foolproof
        """
        p_dict = self.state_dict()
        for p, thet in zip(p_dict, theta):
            p_dict[p] = thet.data
        self.load_state_dict(p_dict)
        self.eval()
        self.train()  # supposed to be important when you set parameters or load state
        
class SirenNetDelta(SirenNet):
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        
        pred = z + y

        loss = torch.sqrt(F.mse_loss(pred, y))

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x) + y
        
#MLP
class MLP(pl.LightningModule):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, activation='ReLU', lr=0.0001) -> None:
        super().__init__()
        self.dim_in = dim_in 
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers 
        self.activation = getattr(torch.nn, activation)
        self.lr = lr
        
        layers = []
        
        for i in range(self.num_layers):
            layers.append(nn.Linear(in_features=self.dim_in if i == 0 else self.dim_hidden, out_features=self.dim_out if i == (self.num_layers - 1) else self.dim_hidden, bias=True))
            layers.append(self.activation())
            
        self.layers = nn.Sequential(*layers)
            
    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)

        loss = torch.sqrt(F.mse_loss(z, y))

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer 
    
class MLPDelta(MLP):
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        
        pred = z + y

        loss = torch.sqrt(F.mse_loss(pred, y))

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x) + y
          
                       
config = BaseConfig()
with open("config/hash_config.json") as f:
    enco_config = json.load(f)
    
model = MLP(dim_in=config.dim_in, dim_hidden=config.dim_hidden, dim_out=config.dim_out, num_layers=config.num_layers, lr=config.lr)              

mri_path = 'data/equinus_downsampled.nii.gz'
mri_path = 'data/cube.nii.gz'
stl_path = 'output.stl'
label = np.zeros((nib.load(mri_path).shape))

cube = np.ones((128, 128, 128), dtype=np.float32)
cube[:32, :, :] = 0
cube[96:, :, :] = 0

cube[:, :32, :] = 0
cube[:, 96:, :] = 0

cube[:, :, :32] = 0
cube[:, :, 96:] = 0

nib.save(nib.Nifti1Image(cube, affine=np.eye(4)), 'data/cube.nii.gz')

#alternative function using only trimesh
def nii_2_mesh(mri_path, output_path='output.stl'):
    data = nib.load(mri_path).get_fdata(dtype=np.float32)
    #need masking ? Resampling to isotrope also
    data = data / data.max()
    mask = (data > 0.3) * 1.0
    verts, faces, normals, values = measure.marching_cubes(mask, 0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.show()
    
# nii_2_mesh(mri_path)
    
#deforme cube in mesh space
data = nib.load(mri_path).get_fdata(dtype=np.float32)
verts, faces, normals, values = measure.marching_cubes(data, 0)
mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
# mesh.show()

t_range = 10
cube_list = []
for t in range(t_range):
    if t == 0:
        cube_list.append(mesh)
    else:
        verts[:len(verts) // 2] += 10 #create gross deformation
        cube_list.append(trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals))
    
#deform (move?) cube in image space, are indices fixed ? LATER

#create tensordataset. X is (indice, t), Y is (x, y, z)
# indices = torch.linspace(start=0, end=len(verts), steps=len(verts)) #maybe you will have to 0, 1 it
indices = torch.linspace(start=-1 if isinstance(model, SirenNet) else 0, end=1, steps=len(verts)) #maybe you will have to 0, 1 it

times = torch.FloatTensor([])
for t in range(t_range):
    times = torch.cat((times, torch.FloatTensor([t * 0.1]).repeat(len(indices))))

indices = torch.tile(indices, dims=(10,))

if config.dim_in == 4:
    times = times.unsqueeze(-1) #remove if using indices

coords = torch.FloatTensor([])
for cube in cube_list:
    coords = torch.cat((coords, torch.FloatTensor(cube.vertices)))
    
coords = (coords - coords.min()) / (coords.max() - coords.min())
if isinstance(model, SirenNet):
    coords = coords * 2 - 1
    
if config.dim_in == 2:
    X = torch.stack((indices, times), dim=-1)
else:
    X = torch.cat((coords, times), dim=-1)

Y = torch.FloatTensor([])
for i in range(t_range):
    # Y = torch.cat((Y, torch.FloatTensor(cube_list[i].vertices / 186))) 
    Y = torch.cat((Y, torch.FloatTensor(cube_list[i].vertices))) 
    
Y = (Y - Y.min()) / (Y.max() - Y.min())
if isinstance(model, SirenNet):
    Y = Y * 2 - 1

#F(idx, t) = (x, y, z)
dataset = TensorDataset(X, Y)

train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=os.cpu_count() // 4)

test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=os.cpu_count() // 4)  
                
trainer = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
    precision=16,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)
# trainer = pl.Trainer(gpus=config.device, max_epochs=config.epochs)
trainer.fit(model, train_loader)

#create a prediction
pred = torch.concat(trainer.predict(model, test_loader))

for i in range(t_range):
    pred_i = pred[24576 * i:24576 * (i + 1)] #first cube
    predmesh = trimesh.Trimesh(vertices=pred_i.detach().cpu().numpy(), faces=faces, vertex_normals=normals)
    predmesh.export(model.logger.log_dir + '/' + f'pred{i}.stl')
