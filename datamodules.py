"""
Datamodules for MRI interpolation. Each module comes with an optional 2D MNIST setup for tests purposes. Not all datamodules are compatible with all networks (implicit representation vs conventional voxel view)
Data options:
-MNIST for tests
-DHCP for real deal
"""
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, TensorDataset


def display_output(batch):
    """
    Small KOL function for visualising MNIST
    Args:
        batch(tuple of tensors): x is coordinates, y is intensity
    Returns:
        None, visu only
    """
    x, y = batch
    x = x.squeeze()
    y = y.squeeze()
    plt.imshow(y.reshape(28, 28))
    plt.show()
    plt.clf()
    return None


class MNISTDataModule(pl.LightningDataModule):
    """
    Datamodule for implicit representation training based on MNIST dataset
    loader returns coordinates (-1, 1) as x and intensity as y
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config

    def prepare_data(self) -> None:
        """
        Steps to be done on 1 GPU, like datafetch
        """
        self.mnist_dataset = torchvision.datasets.MNIST(
            root=self.config.dataset_path, download=False
        )
        return super().prepare_data()

    def setup(self) -> None:
        """
        steps to be done on multiple GPUs, like datatransforms
        """
        if self.config.initialization == "mean":
            # get all digits from train_targets and mean them
            digit_tensor = torch.zeros(28, 28)
            for digit_idx, target in enumerate(self.mnist_dataset.targets):
                if int(target) in self.config.train_target:
                    digit_tensor += torchvision.transforms.ToTensor()(
                        self.mnist_dataset[digit_idx][0]
                    ).squeeze()

        # fetch the wanted train digit, 1 digit version
        if self.config.initialization == "single":
            for digit_idx, target in enumerate(self.mnist_dataset.targets):
                if int(target) in self.config.train_target:
                    digit_tensor = torchvision.transforms.ToTensor()(
                        self.mnist_dataset[digit_idx][0]
                    ).squeeze()
                    break

        if self.config.initialization == "random":
            targets_list = []
            for digit_idx, target in enumerate(self.mnist_dataset.targets):
                if int(target) in self.config.train_target:
                    targets_list.append(self.mnist_dataset[digit_idx][0])
            digit_tensor = torch.empty(0)
            for idx in range(len(targets_list)):
                target_tensor = torchvision.transforms.ToTensor()(
                    targets_list[idx]
                ).squeeze()
                flat = torch.Tensor(target_tensor.flatten()).unsqueeze(-1)
                digit_tensor = torch.cat((digit_tensor, flat))

        # normalization
        digit_tensor = digit_tensor * 2 - 1
        digit_shape = [28, 28]  # hardcoded MNIST

        x = torch.linspace(-1, 1, digit_shape[0])
        y = torch.linspace(-1, 1, digit_shape[1])
        mgrid = torch.stack(torch.meshgrid(x, y), dim=-1)

        if self.config.initialization == "random":
            x_flat = torch.Tensor(mgrid.reshape(-1, 2)).repeat(len(targets_list), 1)
            y_flat = digit_tensor
        else:
            x_flat = torch.Tensor(mgrid.reshape(-1, 2))
            y_flat = torch.Tensor(digit_tensor.flatten()).unsqueeze(-1)

        self.dataset = torch.utils.data.TensorDataset(x_flat, y_flat)

        return None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None


class MriImage(Dataset):
    """
    Dataset for implicit representation training.
    coordinates are returned in x
    Intensity is returned in y
    """

    def __init__(
        self, config, image_path: str = None, norm_siren: bool = False, *args, **kwargs
    ):
        super().__init__()
        if image_path:
            image = nib.load(image_path)
        else:
            image = nib.load(config.image_path)
        image = image.get_fdata(dtype=np.float32)

        axes = []
        if norm_siren:
            for s in image.shape:
                axes.append(torch.linspace(-1, 1, s))
        else:
            for s in image.shape:
                axes.append(torch.linspace(0, 1, s))

        mgrid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)

        # create data tensors
        pixels = torch.FloatTensor(image)
        pixels = pixels.flatten()
        # normalisation
        if norm_siren:
            pixels = (
                (pixels - torch.min(pixels)) / (torch.max(pixels) - torch.min(pixels))
            ) * 2 - 1
        else:
            pixels = (pixels - torch.min(pixels)) / (
                torch.max(pixels) - torch.min(pixels)
            )  # * 2 - 1
        coords = torch.FloatTensor(mgrid)
        coords = coords.reshape(len(pixels), config.dim_in)
        assert len(coords) == len(pixels)
        self.coords = coords
        self.pixels = pixels.unsqueeze(-1)

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        return self.coords[idx], self.pixels[idx]


class MriDataModule(pl.LightningDataModule):
    """
    Take ONE mri image and returns coords and pixels, no split on train/val/test for the moment
    """

    def __init__(self, config=None, *args, **kwargs):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.config = config

    def prepare_data(self) -> None:
        # self.dataset = MriImage(config=self.config)
        self.dataset = MriImage(config=self.config)
        # self.mean_dataset = MriImage(config=self.config, image_path='data/mean.nii.gz') #how to set mean without screwing up ?
        self.train_ds = self.dataset
        self.test_ds = self.dataset
        self.val_ds = self.dataset

    def setup(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,  # true allow reasonnable results with small batch size
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )

    def mean_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mean_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def upsampling(self, shape, batch_size, norm_siren: False):
        """
        Returns a mock loader for upsampling using implicit representation
        """
        axes = []
        if norm_siren:
            for s in shape:
                axes.append(torch.linspace(-1, 1, s))
        else:
            for s in shape:
                axes.append(torch.linspace(0, 1, s))

        mgrid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        fake_pix = torch.zeros(np.product(shape))
        coords = torch.FloatTensor(mgrid)
        coords = coords.reshape(len(fake_pix), len(shape))
        assert len(coords) == len(fake_pix)
        fake_pix = fake_pix.unsqueeze(-1)
        return DataLoader(
            TensorDataset(coords, fake_pix),
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )


# multi frame dataset et datamodule, for MultiHash especially
class MriFrames(Dataset):
    """
    Dataset for implicit representation training.
    coordinates are returned in x
    Intensity is returned in y

    One batch is one frame, as batches need to be labeled with the proper index

    """

    def __init__(self, config, image_path=None, *args, **kwargs):
        super().__init__()
        if image_path:
            image = nib.load(image_path)
        else:
            image = nib.load(config.image_path)
        self.image = image.get_fdata(dtype=np.float32)  # [64:192, 64:192, 100:164]

        axes = []
        for s in self.image.shape:
            axes.append(torch.linspace(0, 1, s))

        mgrid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        # create data tensors
        pixels = torch.FloatTensor(self.image)
        pixels = pixels.reshape(-1, self.image.shape[-1], 1)
        # normalisation, should be recasted with torch reshape func
        pixels = (
            (pixels - torch.min(pixels)) / torch.max(pixels)
        ) * 2 - 1  # pixels = pixels.reshpae(pixels, (-1, 1)) should also work
        coords = torch.FloatTensor(mgrid)
        coords = coords.reshape(len(pixels), config.dim_in, self.image.shape[-1])
        assert len(coords) == len(pixels)
        self.coords = coords
        self.pixels = pixels

    def __len__(self):
        return self.image.shape[-1]

    def __getitem__(self, idx):
        return self.coords[:, :, idx], self.pixels[:, idx], idx


class MockMriFrames(Dataset):
    """
    Moch dataset used for data upsampling after training
    """

    def __init__(self, config, shape, *args, **kwargs):
        super().__init__()
        self.shape = shape
        axes = []
        for s in shape:
            axes.append(torch.linspace(0, 1, s))

        mgrid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        # create data tensors
        # pixels = torch.FloatTensor(self.image)
        pixels = torch.ones(np.prod(shape))
        pixels = pixels.reshape(-1, shape[-1], 1)

        coords = torch.FloatTensor(mgrid)
        coords = coords.reshape(len(pixels), config.dim_in, shape[-1])
        assert len(coords) == len(pixels)
        self.coords = coords
        self.pixels = pixels

    def __len__(self):
        return self.shape[-1]

    def __getitem__(self, idx):
        return self.coords[:, :, idx], self.pixels[:, idx], idx


class MriFramesDataModule(pl.LightningDataModule):
    """
    Take ONE mri image and returns coords and pixels, no split on train/val/test for the moment
    """

    def __init__(self, config=None, *args, **kwargs):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.config = config

    def prepare_data(self) -> None:
        # self.dataset = MriImage(config=self.config)
        self.dataset = MriFrames(config=self.config)
        # self.mean_dataset = MriImage(config=self.config, image_path='data/mean.nii.gz') #how to set mean without screwing up ?
        self.train_ds = self.dataset
        self.test_ds = self.dataset
        self.val_ds = self.dataset

    def setup(self, normalisation: str = "zero centered"):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )

    def mean_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mean_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def upsampling(self, shape):
        """
        Returns a mock loader for upsampling using implicit representation
        """
        upsampled_ds = MockMriFrames(config=self.config, shape=shape)
        return DataLoader(
            upsampled_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
