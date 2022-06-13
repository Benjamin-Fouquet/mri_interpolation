"""
Tentative minimal implementation of Varnet and co
TODO:
-remove need for setup after instantation
"""
from dataclasses import dataclass
from sys import path_importer_cache
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn

import nibabel as nb
import torchio as tio
from models import ThreeDCNN
from mri_dataloading import MriDataModule, tensor_visualisation
from pyexpat import model


# Dataclass for hyperparameters TODO: interface with hydra
@dataclass
class Hyperparameters:
    phi_channels: list
    phi_lr: float
    epochs: int
    batch_size: int
    lr: float


class LitModel(pl.LightningModule):
    def __init__(
        self,
        num_channels: List[int] = (128, 128),
        kernel_size: int = 3,
        activation_func: str = "ReLU",
        xp_parameters=None,
        logging: bool = False,
        lr=0.001,
        ckpt_path: str = "data/epoch=19-step=159.ckpt",  # CNN 128 128 50p 20 epochs checkpoint
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
        self.lambda1 = 0.9  # gt weigthing for loss
        self.lambda2 = 0.1  # phi prior weighting for loss

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

        # import prior and checkpoint for prediction
        phi = ThreeDCNN(
            num_channels=self.num_channels
        )  # TODO: bugSomehow numchannels is not taken into account, has to be set at models level. Also, coupling
        self.phi_model = phi.load_from_checkpoint(
            ckpt_path, strict=False
        )  # map_location=gpus needed ?

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def loss(self, y_pred, phi_pred, mask, y) -> torch.Tensor:
        mse_loss = nn.MSELoss()
        mask_inv = torch.ones((mask.shape), device=mask.device) - mask
        phi_loss = mse_loss(y_pred * mask_inv, phi_pred * mask_inv)
        gt_loss = mse_loss(y_pred * mask, y * mask)
        loss = self.lambda1 * gt_loss + self.lambda2 * phi_loss
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> float:
        x, mask, y = train_batch
        y_pred = self.forward(x)
        phi_pred = self.phi_model(
            x
        )  # would call on phi activate gradient graph? If so needs with torhc nograd
        loss = self.loss(y_pred, phi_pred, mask, y)
        if self.logging:
            self.log("train loss: ", loss)
        return loss


# Runner
class MiniRunner:
    def __init__(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        hyperparameters: Hyperparameters,
        gpu=[0],
    ) -> None:
        self.datamodule = datamodule
        self.datamodule.setup()
        self.hyperparameters = hyperparameters
        self.gpu = gpu if torch.cuda.is_available() else []
        # TODO: define phi at this level and feed it into model
        self.model = model
        self.results_path = "results/"  # to put in config, maybe within hydra?

    def setup(self):
        pass

    def train(self, **trainer_kwargs):
        trainer = pl.Trainer(
            gpus=self.gpu, max_epochs=self.hyperparameters.epochs, **trainer_kwargs
        )
        trainer.fit(self.model, train_dataloader=self.datamodule.train_dataloader())
        return self.model, trainer

    def val(self):
        pass

    def test(self):
        pass

    def create_output(
        self, subject: tio.Subject, results_path: str = "results/"
    ) -> None:
        # TODO: change results path, decide for subject or tensor (tensor more flexible ?) or pass directly gt and pred
        file_tag = ""  # TODO: change it later for nice output design
        pred = self.model(subject[0])
        image_pred = pred.detach().numpy().squeeze()
        pred_nii_image = nb.Nifti1Image(image_pred, affine=np.eye(4))
        ground_truth_nii_image = nb.Nifti1Image(
            subject[2].numpy().squeeze(), affine=np.eye(4)
        )
        nb.save(
            img=pred_nii_image, filename=output_path + f"output" + file_tag + ".nii.gz"
        )
        nb.save(
            img=ground_truth_nii_image,
            filename=output_path + f"ground_truth" + file_tag + ".nii.gz",
        )


phi_channels = [128, 128]
phi_lr = 0.001
epochs = 10
lr = 0.001
batch_size = 16
output_path = "results/"
mri_datamodule = MriDataModule()
mri_datamodule.setup()
xp_hyperparameters = Hyperparameters(
    phi_channels=phi_channels,
    phi_lr=phi_lr,
    batch_size=batch_size,
    lr=lr,
    epochs=epochs,
)  # do you really need parameters when pre trained model ?

runner = MiniRunner(
    model=LitModel(), datamodule=mri_datamodule, hyperparameters=xp_hyperparameters
)
runner.setup()
runner.train()
runner.create_output(next(iter(mri_datamodule.train_dataloader())))


print("Done")
