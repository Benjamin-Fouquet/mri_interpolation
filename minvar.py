'''
Tentative minimal implementation of Varnet and co
TODO:
-remove need for setup after instantation
'''
from pyexpat import model
from sys import path_importer_cache
import torch
from torch import nn
from dataclasses import dataclass
from mri_dataloading import MriDataModule
import pytorch_lightning as pl
from models import ThreeDCNN

#Dataclass for hyperparameters
@dataclass
class Hyperparameters:
    phi_channels: list
    phi_lr: float

#Class phi = CNN #needs to be pretrained, so you have to test if you can load it from prior trianing data
class Phi_CNN(nn.Module):
    def __init__(
        self,
        num_channels=(128, 128),
        kernel_size=3,
        activation_func="ReLU",
        lr=0.001,
        *args,
        **kwargs,
                ):
        super().__init__(),
        self.num_channels = num_channels,
        self.kernel_size = 3
        self.activation_func = activation_func
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
    
    def forward(self, x: torch.Tensor):
        return self.model(x)

class LitModel(pl.LightningModule):
    def __init__(
        self,
        num_channels=(128, 128),
        kernel_size=3,
        activation_func="ReLU",
        xp_parameters=None,
        logging=False,
        lr = 0.001,
        ckpt_path='/home/benjamin/Documents/git_repos/mri_interpolation/lightning_logs/version_184/checkpoints/epoch=19-step=1199.ckpt', #placeholder checkpoint
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.activation_func = activation_func
        self.kernel_size = kernel_size
        self.xp_parameters = xp_parameters
        self.logging = logging
        self.losses = []
        self.lr = lr
        self.lambda1 = 0.9 #gt weigthing for loss
        self.lambda2 = 0.1 #phi prior weighting for loss

        ###Main model###
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

        #import prior and checkpoint for prediction
        phi = ThreeDCNN(num_channels=self.num_channels) #TODO: bugSomehow numchannels is not taken into account, has to be set at models level. 
        self.phi_model = phi.load_from_checkpoint(ckpt_path, strict=False) # map_location=gpus needed ?


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def loss(self, y_pred, phi_pred, mask, y) -> float:
        mse_loss = nn.MSELoss()
        phi_loss = mse_loss(y_pred * ~mask, phi_pred * ~ mask)
        gt_loss = mse_loss(y_pred * mask, y * mask)
        loss = self.lambda1 * gt_loss + self.lambda2 * phi_loss
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> float:
        x, mask, y = train_batch
        y_pred = self.forward(x)
        phi_pred = self.phi_model(x) #would call on phi activate gradient graph? If so needs with torhc nograd
        loss = self.loss(y_pred, phi_pred, mask, y)
        if self.logging:
            self.log("train loss: ", loss)
        return loss

#Runner
class MiniRunner:
    def __init__(self, datamodule: pl.LightningDataModule, hyperparameters: Hyperparameters, gpu=2) -> None:
        self.datamodule = datamodule
        self.datamodule.setup()
        self.hyperparameters = hyperparameters
        self.gpu = gpu
        self.model = model
        self.phi = Phi_CNN(num_channels=hyperparameters.phi_channels, lr=hyperparameters.phi_lr) #you may need the checkpoint

    def setup(self):
        pass

    def train(self, **trainer_kwargs):
        trainer = pl.Trainer(gpus=self.gpu, max_epochs=1, **trainer_kwargs)
        trainer.fit(model, train_dataloader=self.datamodule.train_dataloader())
        return model, trainer

    def val(self):
        pass

    def test(self):
        pass

phi_channels = [128, 128]
phi_lr = 0.001
mri_datamodule = MriDataModule()
mri_datamodule.setup()
xp_hyperparameters=Hyperparameters(phi_channels=phi_channels, phi_lr=phi_lr) #do you really need parameters when pre trained model ?

runner = MiniRunner(datamodule=mri_datamodule, hyperparameters=xp_hyperparameters)
runner.setup()
model = LitModel()

print("Done")