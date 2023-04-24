import vtk
import argparse
from scipy.interpolate import griddata, interpn
from torch import autograd as ag
import torch

def nii_2_mesh(filename_nii, filename_stl, label):

    """
    Read a nifti file including a binary map of a segmented organ with label id = label. 
    Convert it to a smoothed mesh of type stl.

    filename_nii     : Input nifti binary map 
    filename_stl     : Output mesh name in stl format
    label            : segmented label id 
    """

    # read the file
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filename_nii)
    reader.Update()
    
    # apply marching cube surface generation
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(reader.GetOutputPort())
    surf.SetValue(0, label) # use surf.GenerateValues function if more than one contour is available in the file	
    surf.Update()
    
    #  auto Orient Normals
    surf_cor = vtk.vtkPolyDataNormals()
    surf_cor.SetInputConnection(surf.GetOutputPort())
    surf_cor.ConsistencyOn()
    surf_cor.AutoOrientNormalsOn()
    surf_cor.SplittingOff()
    surf_cor.Update()
    
    #smoothing the mesh
    smoother= vtk.vtkWindowedSincPolyDataFilter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        smoother.SetInput(surf_cor.GetOutput())
    else:
        smoother.SetInputConnection(surf_cor.GetOutputPort())
    smoother.SetNumberOfIterations(60)
    smoother.NonManifoldSmoothingOn()
    #smoother.NormalizeCoordinatesOn() #The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
    smoother.GenerateErrorScalarsOn()
    smoother.Update()
     
    # save the output
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(smoother.GetOutputPort())
    writer.SetFileTypeToASCII()
    writer.SetFileName(filename_stl)
    writer.Write()

#extract (x, y, z), how many points ? 

t_zero = torch.FloatTensor((0, 0, 0))
t_one = torch.FloatTensor((1, 1, 1))
t_zero.requires_grad = True
t_one.requires_grad = True

delta_t = t_one - t_zero

ag.grad(outputs=delta_t, inputs=t_one)


#extract i

