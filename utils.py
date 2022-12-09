import random as rd
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
# from registration import commonSegment
# from tools import rigidMatrix
from scipy.ndimage import convolve, distance_transform_cdt, map_coordinates

import torchio as tio


def show_slices(
    image: Union[np.ndarray, tio.data.image.ScalarImage, nib.nifti1.Nifti1Image]
) -> None:
    """
    TODO: 2D implementation
    General quick viewer for 3D slices
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


def show(array: np.ndarray) -> None:
    """
    Display real part of passed array
    """
    if np.iscomplexobj(array):
        array = array.real
    if len(array.shape) == 2:
        plt.imshow(array.T, cmap="gray", origin="lower")
        plt.show()
    if len(array.shape) == 3:
        xmean, ymean, zmean = [int(i / 2) for i in array.shape]
        slice_0 = array[xmean, :, :]
        slice_1 = array[:, ymean, :]
        slice_2 = array[:, :, zmean]
        slices = [slice_0, slice_1, slice_2]
        fig, axes = plt.subplots(1, len(slices))
        for i, slice in enumerate(slices):
            axes[i].imshow(slice.T, cmap="gray", origin="lower")
        plt.show()

    return None


def tensor_visualisation(tensor):
    """
    Take a tensor and display a sample of it
    """
    assert torch.is_tensor(tensor), "Data provided is not a torch tensor"
    tensor = tensor.detach().cpu()
    # tensor = tensor.detach().numpy()
    z, y, x = tensor.shape
    fig, axes = plt.subplots(1, len(tensor))
    for i, slice in enumerate(tensor):
        axes[i].imshow(slice.T, origin="lower")  # cmap="gray"
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


def apply_psf(tensor, kernel, image_shape):
    """
    Apply precomputed convolution kernel
    """
    with torch.no_grad():
        image = tensor.reshape(image_shape)
        image = convolve(input=image, weights=kernel, mode="nearest", cval=0)
        image = image.flatten()
        image = image[..., None]
        tensor.data = torch.FloatTensor(image)
    return tensor


def psf_kernel(dim=2):
    """
    return 5 x 5 (x 5) gaussian psf kernel
    """
    n_samples = 5
    psf_sx = np.linspace(-0.5, 0.5, n_samples)
    psf_sy = np.linspace(-0.5, 0.5, n_samples)
    psf_sz = np.linspace(-0.5, 0.5, n_samples)

    # Define a set of points for PSF values using meshgrid
    # https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    if dim == 2:
        psf_x, psf_y = np.meshgrid(psf_sx, psf_sy, indexing="ij")
    if dim == 3:
        psf_x, psf_y, psf_z = np.meshgrid(psf_sx, psf_sy, psf_sz, indexing="ij")

    # Define gaussian kernel as PSF model
    sigma = (
        1.0 / 2.3548
    )  # could be anisotropic to reflect MRI sequences (see Kainz et al.)

    def gaussian(x, sigma):
        return np.exp(-x * x / (2 * sigma * sigma))

    if dim == 2:
        psf = gaussian(psf_x, sigma) * gaussian(psf_y, sigma)
        psf = psf / np.sum(psf)
    if dim == 3:
        psf = gaussian(psf_x, sigma) * gaussian(psf_y, sigma) * gaussian(psf_z, sigma)
        psf = psf / np.sum(psf)

    return psf


def expend_x(tensor, data):
    """
    Returns expended tensor followinf PSF (5 x 5), edges are ignored to stay within the -1, 1 boundaries
    """
    x = torch.linspace(-1, 1, steps=data.shape[0] * 5 - 2)
    y = torch.linspace(-1, 1, steps=data.shape[1] * 5 - 2)
    mgrid = torch.stack(torch.meshgrid(x, y), dim=-1)
    coords = mgrid.reshape(int(len(x) * len(y)), 2)
    return coords


def psf_sum(y_pred, psf, data):
    """
    TODO: finish psf_sum
    """
    y_pred = y_pred.reshape(data.shape[0] * 5 - 2, data.shape[1] * 5 - 2)
    y_sum = torch.zeros(data.shape[0:2])
