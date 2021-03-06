import torchio as tio
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

import matplotlib.pyplot as plt


def tensor_visualisation(tensor):
    """
    Take a tensor and display a sample of it
    """
    assert torch.is_tensor(tensor), "Data provided is not a torch tensor"
    tensor = tensor.detach().cpu().squeeze()
    # tensor = tensor.detach().numpy()
    z, y, x = tensor.shape
    fig, axes = plt.subplots(1, len(tensor))
    for i, slice in enumerate(tensor):
        axes[i].imshow(slice.T, origin="lower")  # cmap="gray"
        if i == 5:
            break
    plt.savefig("debug.png")


def create_rn_mask(subject, percentage):
    shape = subject.t2.shape
    rn_mask = torch.FloatTensor(
        np.random.choice(
            [1, 0], size=shape, p=[(percentage * 0.01), 1 - ((percentage * 0.01))]
        )
    )
    undersampling = rn_mask * subject.t2.data
    subject.add_image(tio.LabelMap(tensor=rn_mask, affine=subject.t2.affine), "rn_mask")
    subject.add_image(
        tio.ScalarImage(tensor=undersampling, affine=subject.t2.affine), "rn_t2"
    )
    return None


class MriDataset(Dataset):
    def __init__(self, subject_ds, *args, **kwargs):
        self.subject_ds = subject_ds

    def __len__(self):
        return len(self.subject_ds)

    def __getitem__(self, idx):
        subject_item = self.subject_ds[idx]
        obs_mask_item = subject_item["rn_mask"].data  # coupling
        obs_item = subject_item["rn_t2"].data
        gt_item = subject_item["t2"].data
        return obs_item, obs_mask_item, gt_item


class MriDataModule(pl.LightningDataModule):
    def __init__(
        self,
        mri_path="data/HCP/",
        patch_size=(256, 256, 32),
        patch_overlap=(16, 16, 16),
        percentage=50,
        *args,
        **kwargs
    ):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.mri_path = mri_path
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.percentage = percentage

    def setup(self):

        subjects = []
        subject = tio.Subject(
            t2=tio.ScalarImage(self.mri_path + "100307_T2.nii.gz"),
            label=tio.LabelMap(self.mri_path + "100307_mask.nii.gz"),
        )

        transforms = [
            tio.RescaleIntensity(out_min_max=(0, 1)),
            # tio.RandomAffine(),
        ]

        subjects = [subject]

        for subject in subjects:
            create_rn_mask(subject, percentage=self.percentage)

        transform = tio.Compose(transforms)

        subjects_dataset = tio.SubjectsDataset(subjects, transform=transform)

        # TODO: later, split with random_split from torch, patch at this level
        if self.patch_size == None:
            mri_dataset = MriDataset(subjects_dataset)
        else:
            # create patch
            grid_sampler = tio.inference.GridSampler(
                subject, self.patch_size, self.patch_overlap
            )
            mri_dataset = MriDataset(grid_sampler)

        self.train_ds = mri_dataset
        self.val_ds = mri_dataset
        self.test_ds = mri_dataset

    def train_dataloader(self):
        return DataLoader(self.train_ds, num_workers=multiprocessing.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.val_ds, num_workers=multiprocessing.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_ds, num_workers=multiprocessing.cpu_count())

    def get_dataset(self):
        return self.train_ds


if __name__ == "__main__":
    mri = MriDataModule()
    mri.setup()
    train_dataloader = mri.train_dataloader()
    data = next(iter(train_dataloader))
