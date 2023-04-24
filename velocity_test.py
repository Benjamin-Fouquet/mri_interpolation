
import trimesh
import numpy as np
import nibabel as nib
from skimage import measure
from torch.utils.data import TensorDataset
import torch

mri_path = 'data/equinus_downsampled.nii.gz'
mri_path = 'data/cube.nii.gz'
stl_path = 'output.stl'
label = np.zeros((nib.load(mri_path).shape))

cube = np.ones((128, 128, 128), dtype=np.float32)
cube[:32, :, :] = 0
cube[96:, :, :] = 0

cube[:, :32, :] = 0
cube[:, 96:, :] = 0

cube[:, :, :32] = 0
cube[:, :, 96:] = 0

nib.save(nib.Nifti1Image(cube, affine=np.eye(4)), 'data/cube.nii.gz')

#alternative function using only trimesh
def nii_2_mesh(mri_path, output_path='output.stl'):
    data = nib.load(mri_path).get_fdata(dtype=np.float32)
    #need masking ? Resampling to isotrope also
    data = data / data.max()
    mask = (data > 0.3) * 1.0
    verts, faces, normals, values = measure.marching_cubes(mask, 0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.show()
    
nii_2_mesh(mri_path)
    
#deforme cube in mesh space
data = nib.load(mri_path).get_fdata(dtype=np.float32)
verts, faces, normals, values = measure.marching_cubes(data, 0)
mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
mesh.show()

t_range = 10
cube_list = [mesh]
for t in range(t_range):
    verts[:len(verts) // 2] += 10 #create gross deformation
    cube_list.append(trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals))
    
#deform (move?) cube in image space, are indices fixed ? LATER

#create tensordataset. X is (indice, t), Y is (x, y, z)
indices = torch.linspace(start=0, end=len(verts), steps=len(verts)) #maybe you will have to 0, 1 it
times = torch.FloatTensor([])
for t in range(t_range):
    times = torch.cat((times, torch.FloatTensor([t]).repeat(len(indices))))

indices = torch.tile(indices, dims=(10,))

indices.unsqueeze(-1)
times.unsqueeze(-1)
X = torch.stack((indices, times), dim=-1)

Y = torch.FloatTensor([])
for i in range(t_range):
    Y = torch.cat((Y, torch.FloatTensor(cube_list[i].vertices)))

#F(idx, t) = (x, y, z)
dataset = TensorDataset(X, Y)