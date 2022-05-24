'''
TODO:
-2D patchs with adaptation and 2D convs
-try SeparableConv2D as drop in replacement for 2D conv
-checkpoints for easy save/load of model
-automated 10% patch overlap
-upscale baboon images before test, to match resolution of training
-make degradation flexible ?
-log for tensorboard at train and val
DONE:
-define train, val, test datasets
-patch structure
'''

import glob
import multiprocessing
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
import torchio as tio
from torch.utils.data import DataLoader, Dataset

from models import ThreeDCNN, ConvModule

batch_size = 32

def show_slices(image):
    """
    TODO: add torchio and numpy compatibility objects compatiblity
    """
    if isinstance(image, nib.nifti1.Nifti1Image):
        data = image.get_fdata()
        data = data.reshape(image.shape[0:3])
    elif isinstance(image, np.ndarray):
        data = image
    elif isinstance(image, tio.data.image.ScalarImage):
        data = image.data.squeeze(0)
    else:
        print("format not recognized")  # TODO: raise exception
    xmean, ymean, zmean = [int(i / 2) for i in data.shape]
    """ Function to display row of image slices """
    slice_0 = data[xmean, :, :]
    slice_1 = data[:, ymean, :]
    slice_2 = data[:, :, zmean]
    slices = [slice_0, slice_1, slice_2]
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.show()

###Lightweight Dataclass
class MriDataset(Dataset):
    '''This class is not needed for dhcp'''
    def __init__(self, subject_ds, *args, **kwargs):
        self.subject_ds = subject_ds

    def __len__(self):
        return len(self.subject_ds)

    def __getitem__(self, idx):
        subject_item = self.subject_ds[idx]
        obs_item = subject_item.t2_degrad.data.squeeze(-1)
        gt_item = subject_item.t2.data.squeeze(-1)
        return obs_item, gt_item

class MriDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dhcp_path: str = "data/DHCP_seg/",
        baboon_path: str ='data/baboon_seg/',
        max_subjects: int = 10,
        patch_size = (64, 64, 1),
        patch_overlap = None, #TODO: automated overlap 10% ?
        *args,
        **kwargs
    ):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.dhcp_path = dhcp_path
        self.baboon_path = baboon_path
        self.max_subjects = max_subjects
        #train/val ~10% hardcoded // potential bug if max subj > number of available data
        self.train_length = int(self.max_subjects * 0.9)
        self.val_length = self.max_subjects - self.train_length
        self.patch_size = patch_size
        self.patch_overlap = (int(size / 10) for size in patch_size) #hardcoded

    def setup(self):
        all_t2s = glob.glob(self.dhcp_path + '*_t2_seg.nii.gz', recursive=True)[:self.max_subjects]
        subjects = []
        # DHCP data for train and val
        for t2_path in all_t2s:
            #normalize
            t2 = tio.RescaleIntensity(out_min_max=(0, 1))(tio.ScalarImage(t2_path))
            t2_degrad = tio.transforms.RandomMotion(degrees=20, translation=0, num_transforms=3, image_interpolation='linear')(t2)
            subjects.append(tio.Subject(t2=t2, t2_degrad=t2_degrad))

down_factor = 3.0
transforms = [
# tio.Resample(
#     (
#         t2.spacing[0] * down_factor,
#         t2.spacing[1] * down_factor,
#         t2.spacing[2] * down_factor,
#     ),
#     image_interpolation="linear",
# ),

tio.transforms.RandomMotion(degrees=8, translation=15, num_transforms=8, image_interpolation='linear'),
tio.Resample(
    (
        t2.spacing[0] * down_factor,
        t2.spacing[1] * down_factor,
        t2.spacing[2] * down_factor,
    ),
    image_interpolation="gaussian",
),
]


t2_degrad = tio.transforms.Compose(transforms=transforms)(t2)

        # for subject in subjects:
            # #Resolution degradation
            # down_factor = 3.0
            # transforms = [
            #     tio.Resample(
            #         (
            #             subject.t2.spacing[0] * down_factor,
            #             subject.t2.spacing[1] * down_factor,
            #             subject.t2.spacing[2] * down_factor,
            #         ),
            #         image_interpolation="linear",
            #     ),
            #     tio.Resample(
            #         (
            #             subject.t2.spacing[0],
            #             subject.t2.spacing[1],
            #             subject.t2.spacing[2],
            #         ),
            #         image_interpolation="nearest",
            #     ),
            # ]

            # Gibbs like effects
            # transforms = [
            #     tio.transforms.RandomAnisotropy(downsampling=(1.5, 5)),
            #     tio.transforms.RandomMotion(degrees=10, translation=10, num_transforms=2, image_interpolation='linear'),
            #     tio.transforms.RandomGhosting(num_ghosts=(4, 10), intensity=(0.5, 1), restore=0.02)
            
            
            # ]

        #test data (use with care)
        all_baboons = glob.glob(self.baboon_path + '*.nii.gz', recursive=True)[:self.max_subjects]
        baboons = []
        for path in all_baboons:
            baboon_image = tio.RescaleIntensity(out_min_max=(0, 1))(tio.ScalarImage(path))
            #TODO: upscale at this level
            baboons.append(tio.Subject(t2_degrad=baboon_image))
        #split
        subjects_dataset = tio.SubjectsDataset(subjects)
        baboons_dataset = tio.SubjectsDataset(baboons)

        if self.patch_size == None:
            self.train_ds, self.val_ds = torch.utils.data.random_split(dataset=subjects_dataset, lengths=[self.train_length, self.val_length])
            self.test_ds = baboons_dataset
        else:
            queue_length = 300
            samples_per_volume = 10
            sampler = tio.data.UniformSampler(patch_size=self.patch_size)
            patches_queue = tio.Queue(
                subjects_dataset=subjects_dataset,
                max_length=queue_length,
                samples_per_volume=samples_per_volume,
                sampler=sampler,
                num_workers=multiprocessing.cpu_count()
            )
            self.train_ds, self.val_ds = torch.utils.data.random_split(dataset=patches_queue, lengths=[self.train_length * samples_per_volume, self.val_length * samples_per_volume])
            self.test_ds = baboons_dataset #TODO: patches on baboon ?

    def train_dataloader(self):
        return DataLoader(self.train_ds, num_workers=multiprocessing.cpu_count() if not self.patch_size else 0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, num_workers=multiprocessing.cpu_count() if not self.patch_size else 0)

    def test_dataloader(self):
        return DataLoader(self.test_ds, num_workers=multiprocessing.cpu_count() if not self.patch_size else 0)

    def get_dataset(self):
        return self.train_ds

DHCP_path = "data/DHCP_seg/"
baboon_path = "data/baboon_seg"
# a dataset
mri_dataloader = MriDataModule(
    mri_path=DHCP_path,
)
mri_dataloader.setup()

batch = next(iter(mri_dataloader.train_dataloader()))

model = ConvModule(input_sample=batch)

trainer = pl.Trainer(gpus=[], max_epochs=1)

trainer.fit(model, train_dataloaders=mri_dataloader.train_dataloader(), val_dataloaders=mri_dataloader.val_dataloader())

# output 1 prediction, try to get it in test loader at a point
baboon_image = nib.load(
    "data/baboon_seg/sub-Formule_ses-03_T2_HASTE_AX2_12_crop_seg.nii.gz"
)
baboon_image = torch.FloatTensor(baboon_image.get_fdata()).squeeze(-1)
pred = model(baboon_image[:, :, 13].unsqueeze(0).unsqueeze(0).unsqueeze(-1)) #TODO: correct the ugly code
image_pred = pred.detach().numpy().squeeze()
pred_nii_image = nib.Nifti1Image(image_pred, affine=np.eye(4))

nib.save(img=pred_nii_image, filename="predicition.nii.gz")