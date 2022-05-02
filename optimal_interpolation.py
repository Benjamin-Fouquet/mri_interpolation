'''
Nibabel interpolation options: https://nipy.org/nibabel/reference/nibabel.processing.html
Skimage interpolation options: https://scikit-image.org/docs/stable/auto_examples/transform/plot_rescale.html

'''

import numpy as np
from scipy import interpolate
import nibabel as nib
from nibabel import processing
import matplotlib.pyplot as plt
import torchio as tio
from skimage.transform import rescale, resize, downscale_local_mean

def show_slices(image):
    if isinstance(image, nib.nifti1.Nifti1Image):
        data = image.get_fdata()
        data = data.reshape(image.shape[0:3])
    if isinstance(image, np.ndarray):
        data = image
    xmean, ymean, zmean = [int(i/2) for i in image.shape[0:3]]
    """ Function to display row of image slices """
    slice_0 = data[xmean, :, :]
    slice_1 = data[:, ymean, :]
    slice_2 = data[:, :, zmean]
    slices = [slice_0, slice_1, slice_2]
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.show()


if __name__ == '__main__':
    #test the function
    path = '/home/benjamin/Documents/Datasets/Baboon/sub-Formule_ses-03_T2_HASTE_AX_9_crop.nii.gz'

    # #Nibabel interpolation
    # img = nib.load(path)
    # show_slices(img)
    # np_image = img.get_fdata()
    # up_factor = 2.0
    # up_voxel_size = [i / up_factor for i in img.header.get_zooms()]
    # out_shape = [i * up_factor for i in img.shape]
    # resample = processing.resample_to_output(in_img=img, voxel_sizes=up_voxel_size, order=3, mode='constant')
    # nib.save(img=resample, filename='output.nii.gz')

    # #torchIO interpolation
    # up_factor = 2.0
    # image = tio.ScalarImage(path)
    # transform = tio.Resample((image.spacing[0] / up_factor, image.spacing[1] / up_factor, image.spacing[2] / up_factor))
    # resample = transform(image)
    # image.save('tio_original.nii.gz')
    # resample.save('tio_resample.nii.gz')

    #sckit-image interpolation
    up_factor = 2.0
    image = nib.load(path)
    np_image = image.get_fdata()
    sci_image = rescale(np_image, 0.25, anti_aliasing=True)
    nib_image = nib.nifti1.Nifti1Image(sci_image, affine=np.eye(4))
    nib.save(nib_image, filename='sci_output.nii.gz')
