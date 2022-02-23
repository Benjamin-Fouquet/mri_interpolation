"""
main for 3D interpolation on mri volumes
For the moment, test on 1 batch and cpu
TODO:
-if Unet, result name fucked
    -find common parameters for both networks
-Add on git for version control, think of structure to keep data outside ?
-The test is performed on cpu, make it at least flexible
-The test can only be performed on batch_size = 1
-Id of subjects is parsed using a ugly split, improvement required
-option for patches, so that it appears in parameters
-2 logs per experiment, one at test one at train...
-pred on fixe subject for the moment
"""
import argparse
import glob
import multiprocessing
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import nibabel as nb
import pytorch_lightning as pl
import torch
import torchio as tio

# local modules
from network import ThreeDCNN, Unet
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A test script")
    parser.add_argument(
        "-i",
        "--input",
        help="Input data",
        type=str,
        required=False,
        default="data/HCP/",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path",
        type=str,
        required=False,
        default="results/",
    )
    parser.add_argument(
        "-p",
        "--patch_size",
        help="Patch size",
        type=int,
        required=False,
        default=(64, 64, 64),
    )
    parser.add_argument(
        "--patch_overlap",
        help="Patch overlap",
        type=int,
        required=False,
        default=(16, 16, 16),
    )
    parser.add_argument(
        "-bs", 
        "--batch_size", 
        help="Batch size", 
        type=int, 
        required=False, 
        default=1
    )
    parser.add_argument(
        "-pct",
        "--percentage",
        help="Percentage of original image used for learning",
        type=int,
        required=False,
        default=50,
    )
    parser.add_argument(
        "-nc",
        "--num_channels",
        help="A list defining hidden layers",
        nargs="+",
        type=int,
        required=False,
        default=[64, 64, 64],
    )
    parser.add_argument(
        "-k",
        "--kernel_size",
        help="size of conv filter",
        type=int,
        required=False,
        default=3,
    )  # make it tuple compatible
    parser.add_argument(
        "-e", 
        "--epochs", 
        help="max epochs", 
        type=int, 
        required=False, 
        default=1
    )
    parser.add_argument(
        "-a",
        "--activation_func",
        help="Activation function",
        type=str,
        required=False,
        default="ReLU",
    )
    parser.add_argument(
        "-l",
        "--logging",
        help="Activate logging functions",
        type=bool,
        required=False,
        default=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="model used for interpolation",
        type=str,
        required=False,
        default="ThreeDCNN",
    )

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    patch_size = args.patch_size
    patch_overlap = args.patch_overlap
    batch_size = args.batch_size
    percentage = args.percentage
    num_channels = args.num_channels
    kernel_size = args.kernel_size
    epochs = args.epochs
    activation_func = args.activation_func
    logging = args.logging
    model = args.model

num_workers = multiprocessing.cpu_count()
device = [2] if torch.cuda.is_available() else []

# parameters for queue
queue_length = 300
samples_per_volume = 10
n_max_subjects = 100

# dictionary for log
xp_parameters = {
    "epochs": epochs,
    "batch_size": batch_size,
    "percentage": percentage,
    "patch size": patch_size,
    "patch_overlap": patch_overlap,
    "channels": num_channels,
    "kernel_size": kernel_size,
    "activation_function": activation_func,
    "queue_length": queue_length,
    "sample_per_volume": samples_per_volume,
    "n_max_subjects": n_max_subjects,
    "test_subject": None,
    "model": model,  # to add
}

subjects = []
n_max_subjects = 100
all_t2s = glob.glob(input_path + "*_T2.nii.gz", recursive=True)[:n_max_subjects]
all_masks = glob.glob(input_path + "*_mask.nii.gz", recursive=True)[:n_max_subjects]

if all_t2s == [] or all_masks == []:
    print("Failed to import data, verify datapath")
    sys.exit()

# loop for subjects
for mask in all_masks:
    id_subject = mask.split("/")[2].split("_")[0:2][
        0
    ]  # ugly hard coded split, TODO: REGEX ?

    t2_file = [s for s in all_t2s if id_subject in s][0]

    subject = tio.Subject(t2=tio.ScalarImage(t2_file), seg=tio.LabelMap(mask))
    subjects.append(subject)


if model == "ThreeDCNN":
    transforms = [
        tio.RescaleIntensity(out_min_max=(0, 1)),
        # tio.transforms.ZNormalization(masking_method='label'),
        # tio.RandomAffine()
    ]
if model == "Unet":
    transforms = [
        tio.RescaleIntensity(out_min_max=(0, 1)),
        #for Unet
        tio.CropOrPad(256) #strong coupling, you can pad for making sure you dont loose data
        # tio.transforms.ZNormalization(masking_method='label'),
        # tio.RandomAffine()
    ]

# mask created before normalisation
def create_rn_mask(subject: tio.Subject, percentage: int) -> None:
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


for subject in subjects:
    create_rn_mask(subject, percentage=percentage)

transform = tio.Compose(transforms)

subjects_dataset = tio.SubjectsDataset(
    subjects, transform=transform
)  # only accept lists

sampler = tio.data.UniformSampler(patch_size)
patches_queue = tio.Queue(
    subjects_dataset=subjects_dataset,
    max_length=queue_length,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=num_workers,
)

patches_loader = DataLoader(
    dataset=patches_queue, batch_size=batch_size, num_workers=0  # must be 0 for patches
)

###
# Grid sampler for sampling accross 1 unique volume
###
# grid_sampler = tio.inference.GridSampler(
#     subject,
#     patch_size,
#     patch_overlap
#     )

# patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size, num_workers=num_workers)
# aggregator = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

dataloader = DataLoader(
    subjects_dataset, num_workers=num_workers, batch_size=1
)  # used for test


if model =="ThreeDCNN":
    network = ThreeDCNN(
        num_channels=num_channels,
        activation_func=activation_func,
        kernel_size=kernel_size,
        xp_parameters=xp_parameters,
        logging=logging,
    )
if model == "Unet":
    network = Unet(
        num_channels=num_channels,
        xp_parameters=xp_parameters,
        logging=logging,   
    )

trainer = pl.Trainer(
    gpus=device, max_epochs=epochs
)  # Trainer(distributed_backend='ddp', gpus=8) for distributed learning

# for advanced profiling
"""
profiler = AdvancedProfiler()
trainer = Trainer(profiler=profiler)
"""

trainer.fit(network, train_dataloader=patches_loader)

# transfert model to cpu for test and prediction
trainer = pl.Trainer(gpus=[])
trainer.test(network, dataloaders=dataloader)

# add version number to output_path
output_path = output_path + str(network.logger.version) + "/"
try:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
except OSError:
    print("Error: Creating directory. " + output_path)

file_tag = f"_{model}_{percentage}_{epochs}_L{num_channels}__{activation_func}"

torch.save(
    network.state_dict(),
    output_path
    + f"state_dict" + file_tag + ".pt",
)

#log losses as image
losses = network.losses
fig = plt.plot(range(len(losses)), losses)
plt.savefig(
    output_path
    + f"loss" + file_tag + ".png"
)
# Create one output image for visusalisation
if isinstance(network, ThreeDCNN):
    pred = network(subject.rn_t2.data.unsqueeze(0))
if isinstance(network, Unet):
    pred = network(subject.rn_t2.data[:, :256, :256, :256].unsqueeze(0)) #TODO: coupling, Unet specific
image_pred = pred.detach().numpy().squeeze()
pred_nii_image = nb.Nifti1Image(image_pred, affine=np.eye(4))
ground_truth_nii_image = nb.Nifti1Image(
    subject.t2.data.detach().numpy().squeeze(), affine=np.eye(4)
)

nb.save(
    pred_nii_image,
    output_path
    + f"output" + file_tag + ".nii.gz",
)
nb.save(
    ground_truth_nii_image,
    output_path
   + f"ground_truth" + file_tag + ".nii.gz",
)


def create_nii(tensor: torch.Tensor, output_name: str) -> None:
    nii_image = nb.Nifti1Image(tensor[0, ...].detach().numpy(), affine=np.eye(4))
    try:
        nb.save(nii_image, output_name + ".nii.gz")
    except Exception as e:
        print(e)
    return None


def patch_visu(patches) -> None:
    """
    Simple patch loader from torchiopatch dataloader :: Not possible to in line matplotlib, you may have to save it
    TODO: solve shape problem

    """
    # create_nii(next(iter(patch_loader))['t2']['data'], 'patch')
    fig, axes = plt.subplot(1, len(patches))
    for i, key in enumerate(patches):
        axes[i].imshow(patches[key]["data"], origin="lower", cmap="gray")
    plt.savefig("debug.png")

    return None
