"""
models for implicit representations
"""
import math
from typing import Any

import commentjson as json
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
# import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
from typing import List, Optional, Union, Tuple
from einops import rearrange
import rff
import encoding
import os

class BaseMLP(pl.LightningModule):
    """
    Fully connected network, base classe for other models
    """

    def __init__(
        self,
        dim_in: int = 2,
        dim_out: int = 1,
        dim_hidden: int = 128,
        n_layers: int = 8,
        activation: torch.nn = nn.ReLU,
        criterion: F = F.mse_loss,
        lr: float = 1e-4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.activation = activation
        self.criterion = criterion
        self.lr = lr

        layers = []
        for i in range(n_layers):
            layers.append(
                nn.Linear(
                    in_features=dim_in if i == 0 else dim_hidden,
                    out_features=dim_out if i == (n_layers - 1) else dim_hidden,
                )
            )
            layers.append(activation())

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x) -> Any:
        return self(x)

    def training_step(self, batch, batch_idx) -> torch.FloatTensor:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.criterion(y, y_pred)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)
    
    def lr_schedulers(self) -> LRSchedulerTypeUnion | List[LRSchedulerTypeUnion] | None:
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer, T_max=10, verbose=True
        )
        return self.scheduler

    # def on_train_end(self) -> None:
    #     writer = SummaryWriter(log_dir=self.logger.log_dir)
    #     writer.add_text(text_string=str(config), tag="configuration")
    #     writer.close()

    def set_parameters(self, theta):
        """
        Manually set parameters using matching theta, not foolproof. Used for meta learning
        """
        p_dict = self.state_dict()
        for p, thet in zip(p_dict, theta):
            p_dict[p] = thet.data
        self.load_state_dict(p_dict)
        self.eval()
        self.train()


# utils for siren
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


class SirenLayer(nn.Module):
    """
    Siren layer
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int = 1,
        w0: float = 30.0,
        sigma: float = 6.0,
        is_first: bool = False,
        use_bias: bool = True,
        activation: nn = None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, sigma=sigma, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, sigma, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(sigma / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out



class SirenNet(BaseMLP):
    """
    PURPOSE:
        Implicit representation of arbitrary functions. Mainly used for 2D, 3D image interpolation
    ATTRIBUTES:
        dim_in: dimension of input
        dim_hidden: dimmension of hidden layers. 128-256 are recommended values
        dim_out: dimension of output
        n_layers: number of layers
        w0: multiplying factor so that f(x) = sin(w0 * x) between layers. Recommended value, 30.0
        w0_initial: see w0, recommended value 30.0 as per paper (See paper 'Implicit Neural Representations with Periodic Activation Functions' sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30)
        use_bias: if bias is used between layers, usually set to True
        final_activation: flexible last activation for tasks other than interpolation. None means identity
        lr: recommended 1e-4
        layers: layers of the model, minus last layer
        last_layer: final layer of model
        losses: list of losses during training
    """

    def __init__(
        self,
        dim_in: int = 3,
        dim_hidden: int = 64,
        dim_out: int = 1,
        n_layers: int = 4,
        w0: float = 30.0,
        w0_initial: float = 30.0,
        sigma: float = 6.0,
        use_bias: bool = True,
        final_activation: nn = None,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dim_hidden = dim_hidden
        self.sigma = sigma
        self.losses = []
        self.lr = lr

        self.layers = nn.ModuleList([])
        for ind in range(n_layers):
            is_first = (
                ind == 0
            )  # change the initialization scheme if the layer is first
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(
                SirenLayer(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    sigma=self.sigma,
                    use_bias=use_bias,
                    is_first=is_first,
                )
            )

        final_activation = (
            nn.Identity() if not exists(final_activation) else final_activation
        )
        self.last_layer = SirenLayer(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            sigma=self.sigma,
            use_bias=use_bias,
            activation=final_activation,
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.last_layer(x)


class Modulator(nn.Module):
    """
    Modulator as per paper 'Modulated periodic activations for generalizable local functional representations'
    """

    def __init__(self, dim_in, dim_hidden, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(n_layers):
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


class ModulatedSirenNet(SirenNet):
    """
    Lightning module for modulated siren. Each layer of the modulation is element-wise multiplied with the corresponding siren layer
    """

    def __init__(
        self,
        dim_in: int = 3,
        dim_hidden: int = 64,
        dim_out: int = 1,
        n_layers: int = 4,
        w0: float = 30.0,
        w0_initial: float = 30.0,
        sigma: float = 6.0,
        use_bias: bool = True,
        final_activation: nn = None,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.w0 = w0
        self.w0_initial = w0_initial
        self.sigma = sigma
        self.use_bias = use_bias
        self.final_activation = final_activation
        self.lr = lr
        self.losses = []

        # networks
        self.modulator = Modulator(
            dim_in=self.dim_in, dim_hidden=self.dim_hidden, n_layers=self.n_layers
        )
        self.siren = SirenNet(
            dim_in=self.dim_in,
            dim_hidden=self.dim_hidden,
            dim_out=self.dim_out,
            n_layers=self.n_layers,
            w0=self.w0,
            w0_initial=self.w0_initial,
            sigma=self.sigma,
            use_bias=self.use_bias,
            final_activation=self.final_activation,
            lr=self.lr,
        )

    def forward(self, x):
        mods = self.modulator(x)

        mods = cast_tuple(mods, self.n_layers)

        for layer, mod in zip(self.siren.layers, mods):

            x = layer(x)

            x *= mod

        return self.siren.last_layer(x)
    

class HashSirenNet(SirenNet):
    """
    Lightning module for modulated siren where the latent encoding fed to the modulator is done via HashEncoding. 
    Each layer of the modulation is element-wise multiplied with the corresponding siren layer
    """

    def __init__(
        self,
        config,
        dim_in: int = 3,
        dim_hidden: int = 64,
        dim_out: int = 1,
        n_layers: int = 4,
        w0: float = 30.0,
        w0_initial: float = 30.0,
        sigma: float = 6.0,
        use_bias: bool = True,
        final_activation: nn = None,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.w0 = w0
        self.w0_initial = w0_initial
        self.sigma = sigma
        self.use_bias = use_bias
        self.final_activation = final_activation
        self.lr = lr
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
            n_layers=self.n_layers,
        )
        self.siren = SirenNet(
            dim_in=self.dim_in,
            dim_hidden=self.dim_hidden,
            dim_out=self.dim_out,
            n_layers=self.n_layers,
            w0=self.w0,
            w0_initial=self.w0_initial,
            use_bias=self.use_bias,
            final_activation=self.final_activation,
            lr=self.lr,
        )

    def forward(self, x):
        lat = self.encoding(x)
        mods = self.modulator(lat)

        mods = cast_tuple(mods, self.n_layers)

        for layer, mod in zip(self.siren.layers, mods):

            x = layer(x)

            x *= mod

        return self.siren.last_layer(x)
    

class PsfSirenNet(SirenNet):
    """
    Psf Siren.
    x is expanded at each step via x_to_psf_x. Optimisation can be done by including most calculation in the init
    psf convolution on y is done via a 1D conv layer. Atm the forward pass is modified so that the conv is done in the batch dimension, with a test. An optimisation is either to pu conv last instead of identity and
    not test each time for layer, or create a dedicated method
    """

    def __init__(
        self,
        dim_in: int = 3,
        dim_hidden: int = 64,
        dim_out: int = 1,
        n_layers: int = 4,
        w0: float = 30.,
        w0_initial: float = 30.,
        use_bias: bool = True,
        final_activation: bool = None,
        lr: float = 1e-4,
        coordinates_spacing: float = None,
        n_sample: int = 5,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.dim_hidden = dim_hidden
        self.lr = lr
        self.n_sample = n_sample

        self.layers = nn.ModuleList([])
        for ind in range(n_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(
                SirenLayer(
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
        self.last_layer = SirenLayer(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
        )
        self.coordinates_spacing = (
            coordinates_spacing if not None else ValueError("No PSF spacing defined")
        )  

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


        for layer in self.layers:

            x = layer(x)

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
    

class RffNet(BaseMLP):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int = 128,
        dim_out: int = 1,
        n_layers: int = 8,
        n_frequencies: int = 128,
        sigma: float = 10.0,
        activation: nn = nn.ReLU,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.activation = activation
        self.lr = lr
        layers = []

        self.encoder = self.encoder = rff.layers.GaussianEncoding(
            sigma=sigma, input_size=dim_in, encoded_size=n_frequencies
        )
        encoding_dim_out = n_frequencies * 2

        for i in range(n_layers):
            layers.append(
                nn.Linear(
                    in_features=encoding_dim_out if i == 0 else dim_hidden,
                    out_features=dim_out if i == (n_layers - 1) else dim_hidden,
                )
            )
            layers.append(activation())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)
        y_pred = self.decoder(z)
        return y_pred
    

class TcnnHashMLP(BaseMLP):
    """
    Model using original tinycudann library
    """    
    def __init__(self, 
                 dim_in: int, 
                 n_levels: int, 
                 n_features_per_level: int,
                 log2_hashmap_size: int,
                 base_resolution: int,
                 per_level_scale: float,
                 interplation_method: str = 'linear', 
                 dim_hidden:int = 64, 
                 dim_out: int = 1,
                 lr: float = 1e-4):
        super().__init__()
        self.dim_in = dim_in
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.interpolation_method = interplation_method
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.lr = lr
        self.latents = [] #used for encoding representation visualisation
        
        self.encoder = tcnn.Encoding(
            n_input_dims=(self.dim_in),
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.n_levels,
                "n_features_per_level": self.n_features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": self.base_resolution,
                "per_level_scale": self.per_level_scale,
                "interpolation": self.interpolation_method,
            },
            dtype=torch.float32,
        )
        
        self.decoder = tcnn.Network(
            n_input_dims=self.encoder.n_output_dims,
            n_output_dims=dim_out,
            network_config={
                "otype": "FullyFusedMLP", 
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.dim_hidden,
                "n_hidden_layers": self.n_layers
                },
        )
        
        
    def forward(self, x):
        z = self.encoder(x)
        y_pred = self.decoder(z)
        return y_pred

    def predict_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        self.latents.append(z)
        y_pred = self.decoder(z)
        return y_pred

    def get_latents(self):
        return self.latents
    

class HashMLP(BaseMLP):
    """
    Model using pure python encoding for hash grids
    """  
    def __init__(self, 
                 dim_in: int, 
                 n_levels: int, 
                 n_features_per_level: int,
                 log2_hashmap_size: int,
                 base_resolution: Tuple[int, ...],
                 finest_resolution: Tuple[int, ...],
                 interplation_method: str = 'linear', 
                 dim_hidden:int = 64, 
                 dim_out: int = 1,
                 activation: nn = nn.GELU, 
                 dropout: float = 0.0,
                 lr: float = 1e-4,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_in = dim_in
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.interpolation_method = interplation_method
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.dropout = dropout
        self.lr = lr
        self.latents = [] #used for encoding representation visualisation
         
        if isinstance(self.base_resolution, int):
                self.encoder = encoding.MultiResHashGrid(
                    dim=self.dim_in,
                    n_levels=self.n_levels,
                    n_features_per_level=self.n_features_per_level,
                    log2_hashmap_size=self.log2_hashmap_size,
                    base_resolution=self.base_resolution,
                    finest_resolution=self.finest_resolution,
                )
        else:
            self.encoder = encoding.MultiResHashGridV2(
                dim=self.dim_in,
                n_levels=self.n_levels,
                n_features_per_level=self.n_features_per_level,
                log2_hashmap_size=self.log2_hashmap_size,
                base_resolution=self.base_resolution,
                finest_resolution=self.finest_resolution,
            )

        self.encoding_dim_out = self.n_levels * self.n_features_per_level

        self.decoder = torch.nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                in_features = self.encoding_dim_out
            else:
                in_features = self.dim_hidden
            block = torch.nn.Sequential(
                # torch.nn.utils.parametrizations.spectral_norm(
                #     torch.nn.Linear(
                #         in_features=in_features,
                #         out_features=self.dim_out
                #         if i == (self.n_layers - 1)
                #         else self.dim_hidden,
                #     ),
                #     n_power_iterations=4,
                #     eps=1e-12,
                #     dim=None,
                # ),
                torch.nn.Linear(in_features=in_features, out_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden),
                torch.nn.BatchNorm1d(
                    num_features=self.dim_out
                    if i == (self.n_layers - 1)
                    else self.dim_hidden
                ),
                activation(),
                torch.nn.Dropout(p=dropout, inplace=False),
            )
            self.decoder.append(block)
         
    def forward(self, x):
        z = self.encoder(x)
        y_pred = self.decoder(z)
        return y_pred

    def predict_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        self.latents.append(z)
        y_pred = self.decoder(z)
        return y_pred

    def get_latents(self):
        return self.latents
      
    
class RealGaborLayer(torch.nn.Module):
    '''
        Implicit representation with Gabor nonlinearity
        
        Inputs;
            dim_in: Input features
            dim_out; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''
    
    def __init__(self, dim_in, dim_out, bias=True,
                 is_first=False, w0=30.0, c=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = w0
        self.scale_0 = c
        self.is_first = is_first
        
        self.dim_in = dim_in
        
        self.freqs = torch.nn.Linear(dim_in, dim_out, bias=bias)
        self.scale = torch.nn.Linear(dim_in, dim_out, bias=bias)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))
    
class ComplexGaborLayer(torch.nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            dim_in: Input features
            dim_out; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    
    def __init__(self, dim_in, dim_out, bias=True,
                 is_first=False, w0=10.0, c=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = w0
        self.scale_0 = c
        self.is_first = is_first
        
        self.dim_in = dim_in
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = torch.nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = torch.nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = torch.nn.Linear(dim_in,
                                dim_out,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())

class GaborNet(pl.LightningModule):
    def __init__(
        self,
        layer_cls,
        dim_in,
        dim_hidden,
        dim_out,
        n_layers,
        sigma,
        w0,
        lr,
        *args,
        **kwargs
    ):
        super().__init__()
        self.layer_cls = layer_cls
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.lr = lr
        self.sigma = sigma
        self.w0 = w0
        
        layers = []
        for i in range(self.n_layers):
            layers.append(self.layer_cls(dim_in=self.dim_in if i == 0 else self.dim_hidden, dim_out=self.dim_out if i == (n_layers -1) else self.dim_hidden, c=self.sigma, w0=self.w0))
            
        self.layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) #weight_decay=1e-5
        return self.optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log("train_loss", loss)
        self.final_loss = float(loss.detach().cpu().numpy()) #parameter used for optuna
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return y_pred
    

class MultiSiren(pl.LightningModule):
    """
    Lightning module for MultiHashMLP. Legacy model
    Batch size = 1 means whole volume, setup this way as you need the frame idx
    """

    def __init__(
        self, dim_in, dim_hidden, dim_out, n_layers, n_frames, lr, *args, **kwargs
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.n_frames = n_frames
        self.lr = lr
        self.losses = []

        self.encoders = nn.ModuleList()
        for _ in range(self.n_frames):
            self.encoders.append(
                SirenNet(
                    dim_in=self.dim_in,
                    dim_hidden=self.dim_hidden,
                    dim_out=self.dim_hidden,
                    n_layers=self.n_layers,
                )
            )
        self.decoder = SirenNet(
            dim_in=self.dim_hidden,
            dim_hidden=self.dim_hidden,
            dim_out=self.dim_out,
            n_layers=self.n_layers,
        )

        self.automatic_optimization = True  # set to False if you need to propagate gradients manually. Usually lightning does a good job at no_grading models not used for a particular training step. Also, grads are not propagated in inctive leaves

    def forward(self, x, frame_idx):
        z = self.encoders[frame_idx](x)
        y_pred = self.decoder(z)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y, frame_idx = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        z = self.encoders[frame_idx](x)  # pred, model(x)
        y_pred = self.decoder(z)
        loss = F.mse_loss(y_pred, y)

        self.losses.append(loss.detach().cpu().numpy())

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        """
        TODO: adapt for frame adaptive.
        """
        x, y, frame_idx = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        z = self.encoders[frame_idx](x)  # pred, model(x)
        y_pred = self.decoder(z)
        return y_pred
    

class MultiHashMLP(pl.LightningModule):
    """
    Lightning module for MultiHashMLP. Legacy model
    Batch size = 1 means whole volume, setup this way as you need the frame idx
    """

    def __init__(self, dim_in, dim_out, n_frames, config, lr, *args, **kwargs):
        super().__init__()
        self.config = config
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_frames = n_frames
        self.lr = lr
        self.losses = []
        self.latents = []

        self.encoders = nn.ModuleList()
        for _ in range(self.n_frames):
            self.encoders.append(
                tcnn.Encoding(n_input_dims=dim_in, encoding_config=config["encoding"])
            )
        self.decoder = tcnn.Network(
            n_input_dims=self.config["encoding"]["n_levels"]
            * self.config["encoding"]["n_features_per_level"],
            n_output_dims=dim_out,
            network_config=config["network"],
        )

        # if torch.cuda.is_available():
        #     self.decoder.to('cuda')

        self.automatic_optimization = True  # set to False if you need to propagate gradients manually. Usually lightning does a good job at no_grading models not used for a particular training step. Also, grads are not propagated in inctive leaves

    def forward(self, x, frame_idx):
        z = self.encoders[frame_idx](x)
        y_pred = self.decoder(z)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y, frame_idx = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        z = self.encoders[frame_idx](x)  # pred, model(x)
        y_pred = self.decoder(z)
        loss = F.mse_loss(y_pred, y)

        self.losses.append(loss.detach().cpu().numpy())

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        """
        TODO: adapt for frame adaptive.
        """
        x, y, frame_idx = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        z = self.encoders[frame_idx](x)
        self.latents.append(z)
        y_pred = self.decoder(z)
        return y_pred

    def get_latents(self):
        return self.latents

# #######
# #TESTS#
# #######

# data = torch.randn(16, 16, 16)

# Y = torch.FloatTensor(data).reshape(-1, 1)
# Y = Y / Y.max()

# axes = []
# for s in data.shape:
#     axes.append(torch.linspace(0, 1, s))

# mgrid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)

# coords = torch.FloatTensor(mgrid)

# X = coords.reshape(len(Y), len(data))

# dataset = torch.utils.data.TensorDataset(X, Y)

# train_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=10000, shuffle=True, num_workers=os.cpu_count()
# )

