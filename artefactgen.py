'''
AIM: procedural generation of artefact data for test in comparison with baboon images

'''
import torchio as tio
import matplotlib.pyplot as plt
from utils import show_slices


baboon_path = "data/baboon_seg/sub-Formule_ses-03_T2_HASTE_AX_9_crop_seg.nii.gz"
dhcp_path = "data/DHCP_seg/sub-CC00060XX03_ses-12501_t2_seg.nii.gz"

baboon_image = tio.RescaleIntensity(out_min_max=(0, 1))(tio.ScalarImage(baboon_path))
dhcp_image = tio.RescaleIntensity(out_min_max=(0, 1))(tio.ScalarImage(dhcp_path))