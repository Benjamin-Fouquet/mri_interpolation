import nibabel as nib 
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
import imageio

log_number = 71
log_path = f'/home/benjamin/results_repaper/version_{log_number}/' 

file = 'interpolation'
im = nib.load(log_path + f'{file}.nii.gz').get_fdata()

dimension = '2D'

if len(im.shape) == 4:
    im = im[:,:,3,:]
    dimension = '3D'


fig, axes = plt.subplots(3, 5)

for j in range(5):
    for i in range(3):
        print(str(j * 3 + i))
        data = im[..., (j * 3 + i)]
        axes[i][j].imshow(data.T, origin="lower", cmap="gray")


with open(f'/home/benjamin/results_repaper/version_{log_number}/config.txt', 'r') as f:
    lines = f.readlines()

config = {}    
for line in lines:
    #remove \n
    line = line[:-1]
    #split before and after :
    key, value = line.split(':')
    key = key[:-1] #remove space
    value = value[1:]
    config[key] = value

enco = config['encoder_type']

title = f'{file}_{enco}_{dimension}' #all attached as used for file names

fig.suptitle(title, fontsize=16)        
plt.savefig(log_path + f'{title}.png')
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
with imageio.get_writer(log_path + f'{title}.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)


        

