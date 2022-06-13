import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift
import matplotlib.pyplot as plt
import nibabel as nib

baboon_path = "data/baboon_seg/sub-Formule_ses-03_T2_HASTE_AX_9_crop_seg.nii.gz"
dhcp_path = "data/DHCP_seg/sub-CC00060XX03_ses-12501_t2_seg.nii.gz"
fabienne_path = 'data/fabienne.nii'

def show(array: np.ndarray) -> None:
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

# image = nib.load(dhcp_path)
# image_np = image.get_fdata()
# image_np = image_np[100,:,:] #dhcp

image = nib.load(baboon_path)
image_np = image.get_fdata()
image_np = image_np[:,:,16] #baboon

#Test how imaginary and real part of fft behave
fourier = fft2(image_np)
fourier = fftshift(fourier)

for i, line in enumerate(fourier.T):
    if i % 2 == 0:
        line[70:270] = line[50:250]
    else: #add negative phase
        line[30:230] = line[50:250]

for i, line in enumerate(fourier):
    if i % 2 == 0:
        line[40:180] = line[30:170]
    else: #add negative phase
        line[20:160] = line[30:170]
    if i == 70:
        break

for i, line in enumerate(fourier.T):
    if i > 70 and i < 130:
        pass
    elif i % 16 == 0:
        line[50:150] = line[30:130]

for i, line in enumerate(fourier):
    line[10:90] = line[20:100]
    if i > 25:
        break

fake = np.zeros((128, 128))

show(np.log(fftshift(fourier)))

show(ifft2(ifftshift(fourier)))

def echo_shift(fourier: np.ndarray, echo_shift: int = 10, percentage: int = 30) -> np.ndarray:
    '''
    shift lines of a fourier space by the shift (voxels) up to the percentage of the fourier space
    '''
    stop = int(fourier.shape[0] * percentage / 100)
    for i, line in enumerate(fourier):
        line[echo_shift:len(line)] = line[0:len(line) - echo_shift]
        if i > stop:
            break
    return fourier

def phase_shift(fourier, phase_shift: int = 30, percentage: int = 30) -> np.ndarray:
    '''
    Shift a portion of the signal from real to imaginary, up to a certain proportion of the fourier space. follows trig rules
    '''
    stop = int(fourier.shape[0] * percentage / 100)
    angle = phase_shift * (np.pi/180)
    for i, line in enumerate(fourier):
        line.real *= np.cos(angle)
        line.imag += np.sin(angle) * line.imag
        if i > stop:
            break
    return fourier


