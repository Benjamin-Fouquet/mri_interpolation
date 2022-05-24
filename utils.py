import numpy as np
import torchio as tio
import nibabel as nib
import matplotlib.pyplot as plt
from typing import Tuple, Union, Dict


def show_slices(image: Union[np.ndarray, tio.data.image.ScalarImage, nib.nifti1.Nifti1Image]) -> None:
    """
    TODO:
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