import nibabel as nib
import glob
import numpy as np
from utils import show_slices

dhcp_path = '/home/benjamin/Documents/Datasets/DHCP/'

all_t2s = glob.glob(dhcp_path + '*_T2w.nii.gz', recursive=True)
all_t1s = glob.glob(dhcp_path + '*_T1w.nii.gz', recursive=True)

t2_mean_image = np.zeros((290, 290, 203))
t1_mean_image = np.zeros((290, 290, 203))

t2_fails = 0
t1_fails = 0

for path in all_t2s:
    image = nib.load(path)
    try:
        t2_mean_image += image.get_fdata()
        print('Success')
    except:
        t2_fails += 1
        print('Fail')
        pass

t2_mean_image /= (len(all_t2s) - t2_fails)
nib.save(nib.Nifti1Image(t2_mean_image, image.affine), dhcp_path + 't2_mean.nii.gz')   

for path in all_t1s:
    image = nib.load(path)
    try:
        t1_mean_image += image.get_fdata()
    except:
        t1_fails += 1
        pass


t1_mean_image /= (len(all_t1s) - t1_fails)
nib.save(nib.Nifti1Image(t1_mean_image, image.affine), dhcp_path + 't1_mean.nii.gz')   

mean_image = (t1_mean_image + t2_mean_image) / 2
nib.save(nib.Nifti1Image(mean_image, image.affine), dhcp_path + 'mean.nii.gz')   


