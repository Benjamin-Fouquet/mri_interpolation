import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import nibabel as nib
import math
import tinycudann as tcnn

batch_size = 20000
epochs = 200
dim_hidden = 352

#Première étape, charger une image IRM
image_path = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'

mri_image = nib.load(image_path)

data = mri_image.get_fdata(dtype=np.float32)
data = data[:,:,3, 7] #commencons en 2D
image_shape = data.shape
dim_in = len(data.shape)

plt.imshow(data.T, origin='lower', cmap='gray')

'''
Reference: Sitzmann et al, Implicit Neural Representations with Periodic Activation Functions

Notebook: https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=39Mf3epV8Ib2
'''

#utilitaires pour Siren
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
        # return (torch.sin(self.w0 * x) + 1) / 2 


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
        w0_initial: see w0, recommended value 30.0 as per paper (See paper 'Implicit Neural Representations with Periodic Activation Functions' sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30)
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
        dim_hidden: int = 352,
        dim_out: int = 1,
        num_layers: int = 4,
        w0=30.0,
        w0_initial=30.0,
        c=6.0,
        use_bias=True,
        final_activation=None,
        lr=1e-4,
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.c = c
        self.losses = []
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
                    c=self.c,
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
            c=self.c,
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

        loss = F.mse_loss(z, y)
        self.losses.append(loss.detach().cpu().numpy())

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
model = SirenNet(
    dim_in=len(data.shape),
    dim_hidden=dim_hidden,
)

'''
Reference: Mehta et al.: Modulated Periodic Activations for Generalizable Local Functional Representations
'''

class Modulator(nn.Module):
    """
    Modulator as per paper 'Modulated periodic activations for generalizable local functional representations'
    """
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(nn.Linear(dim, dim_hidden), nn.GELU()))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim=1)

        return tuple(hiddens)

class ModulatedSirenNet(pl.LightningModule):
    """
    Lightning module for modulated siren. Each layer of the modulation is element-wise multiplied with the corresponding siren layer
    """

    def __init__(
        self,
        dim_in=3,
        dim_hidden=352,
        dim_out=1,
        num_layers=4,
        w0=30.0,
        w0_initial=30.0,
        c=6.0,
        use_bias=True,
        final_activation=None,
        lr=1e-4,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.w0 = w0
        self.w0_initial = w0_initial
        self.c = c
        self.use_bias = use_bias
        self.final_activation = final_activation
        self.lr = lr
        self.losses = []

        # networks
        self.modulator = Modulator(
            dim_in=self.dim_in, dim_hidden=self.dim_hidden, num_layers=self.num_layers
        )
        self.siren = SirenNet(
            dim_in=self.dim_in,
            dim_hidden=self.dim_hidden,
            dim_out=self.dim_out,
            num_layers=self.num_layers,
            w0=self.w0,
            w0_initial=self.w0_initial,
            c=self.c,
            use_bias=self.use_bias,
            final_activation=self.final_activation,
            lr=self.lr,
        )

    def forward(self, x):
        mods = self.modulator(x)

        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.siren.layers, mods):

            x = layer(x)

            x *= mod

        return self.siren.last_layer(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

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
    


###################
#Passons en 2D + t#
###################

batch_size = 30000
epochs = 50

#On augmente la dimension de l'image
mri_image = nib.load(image_path)

data = mri_image.get_fdata(dtype=np.float32)
data = data[:,:,3,:] #largeur, hauteur, nombre de slices, nombre de frames
image_shape = data.shape
dim_in = len(data.shape)
#data.transpose()

Y = torch.FloatTensor(data).reshape(-1, 1)
Y = Y / Y.max() #* 2 -1

axes = []
for s in image_shape:
    axes.append(torch.linspace(0, 1, s))

mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

coords = torch.FloatTensor(mgrid)

X = coords.reshape(len(Y), dim_in)

dataset = torch.utils.data.TensorDataset(X, Y)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())



# Y = torch.FloatTensor(data).reshape(-1, 1)
# Y = Y / Y.max()

# axes = []
# for s in image_shape:
#     axes.append(torch.linspace(0, 1, s))

# mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

# coords = torch.FloatTensor(mgrid)

# X = coords.reshape(len(Y), dim_in)

# dataset = torch.utils.data.TensorDataset(X, Y)

# train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

# test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

model = ModulatedSirenNet(
    dim_in=len(data.shape),
    dim_hidden=dim_hidden,
)

trainer = pl.Trainer(
    gpus=[0] if torch.cuda.is_available() else [],
    max_epochs=epochs,
    precision=32,
)

trainer.fit(model, train_loader)
pred = torch.concat(trainer.predict(model, test_loader))

im = pred.reshape(data.shape)
im = im.detach().cpu().numpy()
im = np.array(im, dtype=np.float32)

im = im[..., 7]

gt = data[..., 7]
gt = gt /gt.max() * 2 - 1

diff = im - gt

fig, axes = plt.subplots(1, 3)

axes[0].imshow(gt.T, origin='lower', cmap='gray')
axes[1].imshow(im.T, origin='lower', cmap='gray')
axes[2].imshow(diff.T, origin='lower', cmap='gray')
     
plt.savefig('output.png')