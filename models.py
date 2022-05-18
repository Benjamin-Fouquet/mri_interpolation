"""
Models
TODO:
-Validation step in models
DONE:
-Loggs
-losses
"""

from dataclasses import dataclass
from typing import Tuple, Union, Dict, Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchio as tio
import torchvision
from torch.utils.data import Dataset, DataLoader
from skimage import metrics  # mean_squared_error, peak_signal_noise_ratio
import matplotlib.pyplot as plt
import os


class ConvModule(pl.LightningModule):
    '''
    Base conv module for quick prototyping, including logging and QOL
    TODO: logging
    '''
    def __init__(
        self,
        channels: Tuple[int, ...] = [128, 128, 128],
        input_sample: Any = None, #a batch of dataloader, None assume classic torch tensor batch, necessary for full logging
        learning_rate: float = 0.001,
        kernel_size: Union[int, Tuple[int, ...]] =3,
        activation_func: str = 'ReLU',
        *args,
        **kwargs, 
    ):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.learning_rate = learning_rate
        self.logging = False
        self.input_sample = input_sample
        #if torchio dataset, adapt network to accept torchio keys
        if isinstance(self.input_sample, dict):
            self.set_type = 'torchio'
            keys = []
            for key in input_sample:
                keys.append(key)
            self.x_key = keys[0]
            self.y_key = keys[1]

        #2D or 3D conv
        if self.set_type == 'torchio':
            batch_shape = input_sample[self.x_key]['data'].shape
        else:
            batch_shape = input_sample[0].shape
        
        if len(batch_shape) == 5:
            conv_layer = nn.Conv3d
        elif len(batch_shape) == 4:
            conv_layer = nn.Conv2d

        #activation function
        if activation_func == "Tanh":
            activation_layer=nn.Tanh()
        if activation_func == "ReLU":
            activation_layer=nn.ReLU()
        if activation_func == "Sig":
            activation_layer=nn.Sigmoid()
        
        #Build the layer system
        layers = []
        for idx in range(len(channels)):
            in_channels = channels[idx - 1] if idx > 0 else 1
            out_channels = channels[idx]
            layer = conv_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            )

            layers.append(layer)
            layers.append(activation_layer)

        last_layer = conv_layer(
            in_channels=channels[-1],
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )
        layers.append(last_layer)
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def loss(self, y_pred:torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(y_pred, y)

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def log_parameters(self) -> None:
        if self.xp_parameters != None:
            txt_log = ""
            for key, value in enumerate(self.xp_parameters):
                txt_log += f"{key}: {value}"
                txt_log += "\n"
            self.logger.experiment.add_text("Data", txt_log)

        # log of image, gt and difference before converting to np

        if self.set_type == 'torchio':
            x, y = (
                self.input_sample[self.x_key]['data'],
                self.input_sample[self.x_key]['data'],
            )
        else:
            x, y = (
                self.input_sample[0]['data'],
                self.input_sample[1]['data'],
            )
        if self.logging and self.input_sample != None:

            y_pred = self.forward(x)
            diff = y - y_pred
            ground_truth_grid = torchvision.utils.make_grid(
                y[..., int(y.shape[-1] / 2)]
            )  # slicing around middle
            self.logger.experiment.add_image("ground-truth", ground_truth_grid, 0)
            pred_grid = torchvision.utils.make_grid(
                y_pred[..., int(y.shape[-1] / 2)]
            )  # slicing around middle
            self.logger.experiment.add_image("prediction", pred_grid, 0)
            diff_grid = torchvision.utils.make_grid(
                diff[..., int(y.shape[-1] / 2)]
            )  # slicing around middle
            self.logger.experiment.add_image("differences", diff_grid, 0)

            # detach and log metrics
            y = y.detach().numpy().squeeze()
            y_pred = y_pred.detach().numpy().squeeze()
            self.log("MSE", metrics.mean_squared_error(y, y_pred))
            self.log("PSNR", metrics.peak_signal_noise_ratio(y, y_pred))
            self.log("SSMI", metrics.structural_similarity(y, y_pred))
        
        return None

    def training_step(self, batch, batch_idx) -> float:
        if self.set_type == 'torchio':
            x, y = batch[self.x_key]["data"], batch[self.y_key]["data"]
        else:
            x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.train_losses.append(loss.detach().cpu().numpy())
        if self.logging:
            self.log("train loss: ", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> float:
        if self.set_type == 'torchio':
            x, y = batch[self.x_key]["data"], batch[self.y_key]["data"]
        else:
            x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.val_losses.append(loss.detach().cpu().numpy())
        if self.logging:
            self.log("val loss: ", loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        pass

    def on_fit_end(self) -> None:
        os.mkdir('results' + '/' + str(self.logger.version) + '/')
        #log losses as image TODO: add labels on legend
        fig1 = plt.plot(range(len(self.train_losses)), self.train_losses, color='r', label='train')
        fig2 = plt.plot(range(len(self.val_losses)), self.val_losses, color='g', label='validation') #TODO: repeat val_losses on len of train_loss ?
        plt.savefig('results' + '/' + str(self.logger.version) + '/' + "losses.png")
        if self.logging:
            self.log_parameters()
        return None

###############
# Autoencoder #
###############
class ThreeDCNN(pl.LightningModule):
    def __init__(
        self,
        num_channels=(128, 128),
        kernel_size=3,
        activation_func="ReLU",
        xp_parameters=None,
        logging=False,
        lr = 0.001,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.activation_func = activation_func
        self.kernel_size = kernel_size
        self.xp_parameters = xp_parameters
        self.logging = logging
        self.losses = []
        self.lr = lr

        layers = []
        for idx in range(len(num_channels)):
            in_channels = num_channels[idx - 1] if idx > 0 else 1
            out_channels = num_channels[idx]
            layer = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            )

            layers.append(layer)
            if self.activation_func == "Tanh":
                layers.append(nn.Tanh())
            if self.activation_func == "ReLU":
                layers.append(nn.ReLU())
            if self.activation_func == "Sig":
                layers.append(nn.Sigmoid())
        last_layer = nn.Conv3d(
            in_channels=num_channels[-1],
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )
        layers.append(last_layer)
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def loss(self, y_pred:torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(y_pred, y)

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_parameters(self) -> None:
        if self.xp_parameters != None:
            txt_log = ""
            for key, value in enumerate(self.xp_parameters):
                txt_log += f"{key}: {value}"
                txt_log += "\n"
            self.logger.experiment.add_text("Data", txt_log)
        return None

    def training_step(self, batch, batch_idx) -> float:
        #TODO: reduce coupling:: you can reduce coupling by extracting all variables and ditch ou what you don't need. Or do it at datamodule level
        x, y = batch["rn_t2"]["data"], batch["t2"]["data"]
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.losses.append(loss.detach().cpu().numpy())
        if self.logging:
            self.log("train loss: ", loss)
        return loss

    def validation_step(self, validation_batch, batch_idx):
        pass

    def test_step(self, test_batch, batch_idx):
        """
        Used for logs, does not return a tensor ! Use forward
        Not GPU compatible at the moment
        """
        x, y, mask = (
            test_batch["rn_t2"]["data"],
            test_batch["t2"]["data"],
            test_batch["rn_mask"]["data"],
        )
        y_pred = self.forward(x)

        # log of image, gt and difference before converting to np
        if self.logging:
            diff = y - y_pred
            ground_truth_grid = torchvision.utils.make_grid(
                y[..., int(y.shape[-1] / 2)]
            )  # slicing around middle
            self.logger.experiment.add_image("ground-truth", ground_truth_grid, 0)
            pred_grid = torchvision.utils.make_grid(
                y_pred[..., int(y.shape[-1] / 2)]
            )  # slicing around middle
            self.logger.experiment.add_image("prediction", pred_grid, 0)
            diff_grid = torchvision.utils.make_grid(
                diff[..., int(y.shape[-1] / 2)]
            )  # slicing around middle
            self.logger.experiment.add_image("differences", diff_grid, 0)
            mask_grid = torchvision.utils.make_grid(
                mask[..., int(y.shape[-1] / 2)]
            )  # slicing around middle
            self.logger.experiment.add_image("sampling-mask", mask_grid, 0)

            # detach and log metrics
            y = y.detach().numpy().squeeze()
            y_pred = y_pred.detach().numpy().squeeze()
            self.log("MSE", metrics.mean_squared_error(y, y_pred))
            self.log("PSNR", metrics.peak_signal_noise_ratio(y, y_pred))
            self.log("SSMI", metrics.structural_similarity(y, y_pred))

            # Add parameters
            self.log_parameters()
        return None
        
##########
#  UNET  #
##########
class Unet(pl.LightningModule):
    '''
    Skip connections are concatenated in this version
    Paper reference: https://www.nature.com/articles/s41598-020-59801-x#Sec2
    '''
    def __init__(
        self,
        num_channels=[64, 64, 128, 128, 256, 256, 512],
        down_activation=['ReLU'],
        up_activation=['ReLU'],
        xp_parameters=None,
        logging=False,
        lr = 0.001
    ):
        super().__init__()
        self.num_channels = num_channels
        self.xp_parameters = xp_parameters
        self.logging = logging
        self.losses = []
        self.lr = lr

        layers = []

        #Down layers
        for idx in range(len(self.num_channels)):
            in_channels = self.num_channels[idx - 1] if idx > 0 else 1
            out_channels = self.num_channels[idx]
            layer = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            layers.append(layer)
            #Add activaiton layer, combinable
            for activation in down_activation:
                if activation == "Tanh":
                    layers.append(nn.Tanh())
                if activation == "ReLU":
                    layers.append(nn.ReLU())
                if activation == "Sig":
                    layers.append(nn.Sigmoid())

            #maxpool between down layers
            if idx < (len(self.num_channels) - 1):
                if self.num_channels[idx] != self.num_channels[idx + 1]:
                    layers.append(nn.MaxPool3d(kernel_size=2)) #maxpool paramters?


        #Up layers
        self.num_channels.reverse()
        for idx in range(len(self.num_channels)):
            in_channels = self.num_channels[idx - 1] if idx > 0 else self.num_channels[idx]
            out_channels = self.num_channels[idx]
            if in_channels > out_channels:
                in_channels *= 1.5
            layer = nn.Conv3d(
                in_channels=int(in_channels),
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            layers.append(layer)
            #Add activation layer, combinable
            for activation in up_activation:
                if activation == "Tanh":
                    layers.append(nn.Tanh())
                if activation == "ReLU":
                    layers.append(nn.ReLU())
                if activation == "Sig":
                    layers.append(nn.Sigmoid())

            #upconv between up layers
            if idx < (len(self.num_channels) - 1):
                if self.num_channels[idx] != self.num_channels[idx + 1]:
                    layers.append(nn.Upsample(
                        scale_factor=2, 
                        mode='nearest',
                    )) 

        #final layer, conv and softmax
        last_layer = nn.Conv3d(
            in_channels=self.num_channels[-1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        layers.append(last_layer)
        # layers.append(nn.Softmax())*
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        skip_connections = []
        for i, layer  in enumerate(self.model):
            if i == 0 or i == (len(self.model) - 1): #deal with first and last layer for out of indice
                x = layer.forward(x)
            else:
                #if need save skip = if max pool comes
                if isinstance(self.model[i + 1], nn.MaxPool3d):
                    skip_connections.append(x)
                #if needs cat meaning if previous is upsampling
                if isinstance(self.model[i - 1], nn.Upsample):
                    pop = skip_connections.pop()
                    x = torch.cat((x, pop), 1)
                x = layer.forward(x)
        return x

    def loss(self, y_pred, y):
        return nn.functional.mse_loss(y_pred, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_parameters(self):
        if self.xp_parameters != None:
            txt_log = ""
            for key, value in enumerate(self.xp_parameters):
                txt_log += f"{key}: {value}"
                txt_log += "\n"
            self.logger.experiment.add_text("Data", txt_log)
        return None

    def training_step(self, batch, batch_idx):
        #TODO: decoupling
        x, y = batch["rn_t2"]["data"], batch["t2"]["data"]
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        self.losses.append(loss.detach().cpu().numpy())
        if self.logging:
            self.log("train loss: ", loss)
        return loss

    def validation_step(self, validation_batch, batch_idx):
        pass

    def test_step(self, test_batch, batch_idx):
        """
        Used for logs, does not return a tensor ! Use forward
        Not GPU compatible at the moment
        TODO: decoupling
        """
        x, y, mask = (
            test_batch["rn_t2"]["data"],
            test_batch["t2"]["data"],
            test_batch["rn_mask"]["data"],
        )
        y_pred = self.forward(x)

        # log of image, gt and difference before converting to np
        if self.logging:
            #TODO: all metrics, resize to fit ? ~most probably you will not need to as results will be crap
            diff = y - y_pred
            ground_truth_grid = torchvision.utils.make_grid(
                y[..., int(y.shape[-1] / 2)]
            )  # slicing around middle
            self.logger.experiment.add_image("ground-truth", ground_truth_grid, 0)
            pred_grid = torchvision.utils.make_grid(
                y_pred[..., int(y.shape[-1] / 2)]
            )  # slicing around middle
            self.logger.experiment.add_image("prediction", pred_grid, 0)
            diff_grid = torchvision.utils.make_grid(
                diff[..., int(y.shape[-1] / 2)]
            )  # slicing around middle
            self.logger.experiment.add_image("differences", diff_grid, 0)
            mask_grid = torchvision.utils.make_grid(
                mask[..., int(y.shape[-1] / 2)]
            )  # slicing around middle
            self.logger.experiment.add_image("sampling-mask", mask_grid, 0)

            # detach and log metrics
            y = y.detach().numpy().squeeze()
            y_pred = y_pred.detach().numpy().squeeze()
            self.log("MSE", metrics.mean_squared_error(y, y_pred))
            self.log("PSNR", metrics.peak_signal_noise_ratio(y, y_pred))
            self.log("SSMI", metrics.structural_similarity(y, y_pred))

            # Add parameters
            self.log_parameters()
        return None

#debugging
if __name__ == "__main__":
    #create lightweight dataloader, normat torch and torchIO
    class LitDataset(Dataset):
        def __init__ (self, dim=(128, 128, 128), *args, **kwargs):
            self.x = torch.randn(dim)
            self.y = torch.randn(dim)
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return self.x, self.y

    dataloader = DataLoader(dataset=LitDataset(), batch_size=1, shuffle=False)

    cnn = ThreeDCNN()
    unet = Unet()
    
    print('finished')


    