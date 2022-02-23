"""
Models
TODO:
-Validation step in models
"""

from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from skimage import metrics  # mean_squared_error, peak_signal_noise_ratio


###############
# Autoencoder #
###############
class ThreeDCNN(pl.LightningModule):
    def __init__(
        self,
        num_channels=(3, 7, 7, 3),
        kernel_size=3,
        activation_func="ReLU",
        xp_parameters=None,
        logging=False,
        lr = 0.001
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

    def training_step(self, batch, batch_idx):
        #TODO: reduce coupling
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
    cnn = ThreeDCNN()
    unet = Unet()
    
    print('finished')


    