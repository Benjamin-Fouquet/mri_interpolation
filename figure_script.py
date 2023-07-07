import nibabel as nib 
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
import imageio

log_number = 12
log_path = f'/home/benjamin/results_repaper/version_{log_number}/' 

im = nib.load(log_path + 'interpolation.nii.gz').get_fdata()

if len(im.shape) == 4:
    im = im[:,:,3,:]


fig, axes = plt.subplots(6, 5)

for j in range(5):
    for i in range(6):
        data = im[..., (j * 6 + i)]
        axes[i][j].imshow(data.T, origin="lower", cmap="gray")

#TODO: parse config and get encotype
title = 'Interpolation Hash3D + t'

fig.suptitle(title, fontsize=16)        
plt.savefig(log_path + 'interpolation.png')
plt.clf()


filenames = []

for idx in range(im.shape[-1]):
    plt.imshow(im[..., idx].T, origin="lower", cmap="gray")
    fig.suptitle(title, fontsize=16)   
    filename = f'file_{idx}.png'
    plt.savefig(filename)
    plt.close()
    filenames.append(filename)
    
# build gif
with imageio.get_writer(log_path + 'interpolation.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)


        

