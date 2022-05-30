'''
AIM: procedural generation of artefact data for test in comparison with baboon images

'''
import torchio as tio
import matplotlib.pyplot as plt
from utils import show_slices
import os


baboon_path = "data/baboon_seg/sub-Formule_ses-03_T2_HASTE_AX_9_crop_seg.nii.gz"
dhcp_path = "data/DHCP_seg/sub-CC00060XX03_ses-12501_t2_seg.nii.gz"

baboon_image = tio.RescaleIntensity(out_min_max=(0, 1))(tio.ScalarImage(baboon_path))
dhcp_image = tio.RescaleIntensity(out_min_max=(0, 1))(tio.ScalarImage(dhcp_path))

# baboon_image['data'] = baboon_image['data'][:, :, :, 15].unsqueeze(-1)
# dhcp_image['data'] = dhcp_image['data'][:, :, 100, :].unsqueeze(-1)

down_factor = 3.0

#create a random transformation = create a list of transformation parameters
degrees_list = [5, 10, 15, 20, 30, 40, 50]
translations_list = [0,10, 20]
num_transforms_list = [1, 2, 3, 4, 6, 8, 10]
image_interpolation_list = ['linear', 'nearest', 'gaussian', 'bspline',]

degrad = tio.transforms.RandomMotion(degrees=20, translation=20, num_transforms=4, image_interpolation='linear')(dhcp_image)

show_slices(degrad)


# for translations in translations_list:       
#     for image_interpolation in image_interpolation_list:
#         for num_transforms in num_transforms_list:
#             for degrees in degrees_list:

#                 transforms = [
#                     tio.transforms.RandomMotion(degrees=degrees, translation=translations, num_transforms=num_transforms, image_interpolation=image_interpolation),
#                 ]


#                 image_degrad = tio.transforms.Compose(transforms=transforms)(dhcp_image),
#                 image_lowres = tio.Resample(
#                     (
#                         dhcp_image.spacing[0] * down_factor,
#                         dhcp_image.spacing[1] * down_factor,
#                         dhcp_image.spacing[2] * down_factor,
#                     ),
#                     image_interpolation=image_interpolation,)(image_degrad[0])
                

#                 save_path = f'artefacts_simulation/{degrees}degrees_{translations}translations_{num_transforms}_numTransforms_{image_interpolation}interpolation/'

#                 os.mkdir(save_path)

#                 plt.imshow(dhcp_image['data'].squeeze(0).squeeze(-1).T, cmap="gray", origin="lower")
#                 plt.savefig(save_path + 'dhcp_image.png')

#                 plt.imshow(image_degrad[0]['data'].squeeze(0).squeeze(-1).T, cmap="gray", origin="lower")
#                 plt.savefig(save_path + 'dhcp_degrad.png')

#                 plt.imshow(image_lowres['data'].squeeze(0).squeeze(-1).T, cmap="gray", origin="lower")
#                 plt.savefig(save_path + 'dhcp_lowres.png')

#                 plt.imshow(baboon_image['data'].squeeze(0).squeeze(-1).T, cmap="gray", origin="lower")
#                 plt.savefig(save_path + 'baboon.png')
