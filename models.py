"""
TODO:
-fetch convolutional models
-base class with optimizer and so so that you dont have repeating code
"""
import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import commentjson as json
import tinycudann as tcnn
from einops import rearrange

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


# siren layer
class Siren(nn.Module):
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

        loss = F.mse_loss(z, y)
        self.losses.append(loss.detach().cpu().numpy())

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


class Modulator(nn.Module):
    """
    Modulator as per paper 'Modulated periodic activations for generalizable local functional representations
    """

    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(nn.Linear(dim, dim_hidden), nn.ReLU()))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim=1)

        return tuple(hiddens)


# siren network
class FourrierNet(pl.LightningModule):
    """
    First layer sin combined with MLP ReLU. Does not work
    """
    def __init__(
        self,
        dim_in=3,
        dim_hidden=64,
        dim_out=1,
        num_layers=4,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.losses = []

        self.layers = nn.ModuleList([])
        ###
        for ind in range(num_layers):
            if ind == 0:
                self.layers.append(
                    Siren(
                        dim_in=dim_in,
                        dim_out=dim_hidden,
                        w0=w0_initial,
                        use_bias=use_bias,
                        is_first=ind,
                    )
                )
            else:
                self.layers.append(
                    nn.Linear(in_features=dim_hidden, out_features=dim_hidden)
                )
                self.layers.append(nn.Tanh())

        final_activation = (
            nn.Identity() if not exists(final_activation) else final_activation
        )
        self.last_layer = nn.Linear(in_features=dim_hidden, out_features=dim_out)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr ,weight_decay=1e-5)
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


class PsfSirenNet(SirenNet):
    """
    Psf Siren. 
    x is expanded at each step via x_to_psf_x. Optimisation can be done by including most calculation in the init
    psf convolution on y is done via a 1D conv layer. Atm the forward pass is modified so that the conv is done in the batch dimension, with a test. An optimisation is either to pu conv last instead of identity and
    not test each time for layer, or create a dedicated method 
    """

    def __init__(
        self,
        dim_in=3,
        dim_hidden=64,
        dim_out=1,
        num_layers=4,
        w0=30,
        w0_initial=30,
        use_bias=True,
        final_activation=None,
        lr=0.0001,
        coordinates_spacing=None,
        n_sample=5,
        *args,
        **kwargs
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.losses = []
        self.lr = lr
        self.n_sample = n_sample

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
        self.coordinates_spacing = (
            coordinates_spacing if not None else ValueError("No PSF spacing defined")
        )  # fall back to SirenNet?

        # Build psf coordinates centered around 0
        psf_sx = torch.linspace(
            -coordinates_spacing[0], coordinates_spacing[0], self.n_sample
        )
        psf_sy = torch.linspace(
            -coordinates_spacing[1], coordinates_spacing[1], self.n_sample
        )
        psf_sz = torch.linspace(
            -coordinates_spacing[2], coordinates_spacing[2], self.n_sample
        )

        # Define a set of points for PSF values using meshgrid
        # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        self.psf_coordinates = torch.stack(
            torch.meshgrid(psf_sx, psf_sy, psf_sz), dim=-1
        ).reshape(
            -1, 3
        )  # flatten

        # build the psf weights
        psf_sx = torch.linspace(-0.5, 0.5, self.n_sample)
        psf_sy = torch.linspace(-0.5, 0.5, self.n_sample)
        psf_sz = torch.linspace(-0.5, 0.5, self.n_sample)

        # Define a set of points for PSF values using meshgrid
        # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        psf_x, psf_y, psf_z = torch.meshgrid(psf_sx, psf_sy, psf_sz)

        # Define gaussian kernel as PSF model
        sigma = (
            1.0 / 2.3548
        )  # could be anisotropic to reflect MRI sequences (see Kainz et al.)

        def gaussian(x, sigma):
            return torch.exp(-x * x / (2 * sigma * sigma))

        psf = gaussian(psf_x, sigma) * gaussian(psf_y, sigma) * gaussian(psf_z, sigma)
        psf = psf / torch.sum(psf)
        psf = psf.flatten()
        psf_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=len(psf),
            stride=len(psf),
            padding=0,
            bias=False,
        )
        psf_conv.weight = nn.Parameter(
            psf.unsqueeze(0).unsqueeze(0), requires_grad=False
        )
        self.psf_conv = psf_conv

    def forward(self, x, mods=None):
        mods = cast_tuple(
            mods, self.num_layers
        )  # +1 to account for the conv layer at the end

        for layer, mod in zip(self.layers, mods):

            x = layer(x)

            if exists(mod):  # TODO: could be removed for performance
                x *= rearrange(mod, "d -> () d")

        return self.last_layer(x)

    def x_to_psf_x(self, x: torch.Tensor):
        """
        convert tensor x to the expended version following (5 x 5 x 5) PSF spread
        """
        psf_coordinates = self.psf_coordinates.repeat(len(x), 1).to(x.device)
        x = x.repeat_interleave(self.n_sample * self.n_sample * self.n_sample, 0)
        return x + psf_coordinates

    def training_step(self, batch, batch_idx):
        x, y = batch
        # create psf around x
        x = self.x_to_psf_x(x)

        z = self(x)

        z = self.psf_conv(z.T).T

        loss = F.mse_loss(z, y)
        self.losses.append(loss.detach().cpu().numpy())

        self.log("train_loss", loss)

        return loss


class ModulatedSirenNet(pl.LightningModule):
    """
    Lightning module for modulated siren. Each layer of the modulation is element-wise multiplied with the corresponding siren layer
    TODO: 
    -verify optimizer, confront to normal modulated: done, okay

    """

    def __init__(
        self,
        dim_in=3,
        dim_hidden=64,
        dim_out=1,
        num_layers=4,
        w0=30.0,
        w0_initial=30.0,
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr ,weight_decay=1e-5)
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


class HashSirenNet(pl.LightningModule):
    """
    Lightning module for modulated siren where the latent encoding fed to the modulator is done via HashEncoding. Each layer of the modulation is element-wise multiplied with the corresponding siren layer
    """

    def __init__(
        self,
        dim_in=3,
        dim_hidden=64,
        dim_out=1,
        num_layers=4,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
        lr=1e-4,
        config=None,
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
        self.use_bias = use_bias
        self.final_activation = final_activation
        self.lr = lr
        self.config = config
        self.losses = []

        # networks
        self.encoding = tcnn.Encoding(
            n_input_dims=self.dim_in,
            encoding_config=config["encoding"],
            dtype=torch.float32,
        )
        self.modulator = Modulator(
            dim_in=self.config["encoding"]["n_levels"]
            * self.config["encoding"]["n_features_per_level"],
            dim_hidden=self.dim_hidden,
            num_layers=self.num_layers,
        )
        self.siren = SirenNet(
            dim_in=self.dim_in,
            dim_hidden=self.dim_hidden,
            dim_out=self.dim_out,
            num_layers=self.num_layers,
            w0=self.w0,
            w0_initial=self.w0_initial,
            use_bias=self.use_bias,
            final_activation=self.final_activation,
            lr=self.lr,
        )

    def forward(self, x):
        lat = self.encoding(x)
        mods = self.modulator(lat)

        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.siren.layers, mods):

            x = layer(x)

            x *= mod

        return self.siren.last_layer(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr ,weight_decay=1e-5)
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

class HashMLP(pl.LightningModule):
    '''
    Lightning module for HashMLP
    '''
    def __init__(
        self,
        dim_in,
        dim_out,
        config,
        lr,
        *args,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.lr = lr
        self.losses =[]

        self.encoding = tcnn.Encoding(n_input_dims=dim_in, encoding_config=config['encoding'])
        self.mlp= tcnn.Network(n_input_dims=self.encoding.n_output_dims, n_output_dims=dim_out, network_config=config['network'])
        self.model = torch.nn.Sequential(self.encoding, self.mlp)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr ,weight_decay=1e-5)
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

class MultiHashMLP(pl.LightningModule):
    '''
    Lightning module for MultiHashMLP. Each 
    '''
    def __init__(
        self,
        dim_in,
        dim_out,
        n_frames,
        config,
        lr,
        *args,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_frames = n_frames
        self.lr = lr
        self.losses =[]

        self.encoders = nn.ModuleList()
        for _ in range(self.n_frames):
            self.encoders.append(tcnn.Encoding(n_input_dims=dim_in, encoding_config=config['encoding']))
        self.decoder= tcnn.Network(n_input_dims=self.config["encoding"]["n_levels"] * self.config["encoding"]["n_features_per_level"], n_output_dims=dim_out, network_config=config['network'])


        self.automatic_optimization = False

    def forward(self, x, frame_idx):
        lat =self.encoders[frame_idx](x)
        z = self.decoder(lat)
        return z.float()

    def configure_optimizers(self):
        self.enc_optimizers = []
        for i in range(self.n_frames):
            optimizer = torch.optim.Adam(self.encoders[i].parameters(), lr=self.lr ,weight_decay=1e-5)
            self.enc_optimizers.append(optimizer)

        self.dec_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr ,weight_decay=1e-5)       
        return self.enc_optimizers, self.dec_optimizer 

    def training_step(self, batch, batch_idx):
        x, y, frame_idx = batch
        lat = self.encoders[frame_idx] #pred, model(x)
        z = self.decoder(lat)
        self.enc_optimizers[frame_idx].zero_grad()
        self.dec_optimizer.zero_grad()
        loss = F.mse_loss(z, y)
        self.manual_backward(loss)
        #how to be frame dependant?
        self.enc_optimizers[frame_idx].step()
        self.dec_optimizer.step()

        self.losses.append(loss.detach().cpu().numpy())

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        '''
        TODO: adapt for frame adaptive.
        '''
        x, y = batch
        return self(x)


