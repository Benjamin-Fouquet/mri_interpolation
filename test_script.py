#Nous importons les modules classiques

import torch #bibliothèque de deep learning
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import nibabel as nib
import math
# import tinycudann as tcnn

#Première étape, charger une image IRM
image_path = 'sample_ankle_dyn_mri.nii.gz'
mri_image = nib.load(image_path)

data = mri_image.get_fdata(dtype=np.float32)
data = data[:,:,3, :] #On visualise la 3 slice et la 7eme frame

batch_size = 20000
epochs = 50
dim_hidden = 352

image_shape = data.shape
dim_in = len(data.shape)

#On tensorise nos données puis on normalise entre -1 et 1
Y = torch.FloatTensor(data).reshape(-1, 1)
Y = Y / Y.max()

#Creation de la grille de coordonnées en points d'entrée
axes = []
for s in image_shape:
    axes.append(torch.linspace(0, 1, s))

mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

coords = torch.FloatTensor(mgrid)

X = coords.reshape(len(Y), dim_in) #Je tensorise la grille

dataset = torch.utils.data.TensorDataset(X, Y)  #une coordonnée est associée à une valeur y

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

class MLP(pl.LightningModule):
    def __init__(self, dim_in, dim_hidden, dim_out, n_layers, lr):
        super().__init__()
        self.lr = lr
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_features=dim_in if i == 0 else dim_hidden, out_features=dim_out if i == (n_layers - 1) else dim_hidden))
            layers.append(nn.ReLU())
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)

        loss = F.mse_loss(z, y)
        
        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

model = MLP(dim_in=len(data.shape), dim_hidden=dim_hidden, dim_out=1, n_layers=8, lr=1e-4)

trainer = pl.Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    max_epochs=epochs,
    precision=32, #16 ou 32. 16 pour automated mixed precision, voir https://developer.nvidia.com/automatic-mixed-precision
)

trainer.fit(model, train_loader)

pred = torch.concat(trainer.predict(model, test_loader))

im = pred.reshape(data.shape)
im = im.detach().cpu().numpy()
im = np.array(im, dtype=np.float32)

nib.save(nib.Nifti1Image(im, affine=np.eye(4)), 'pred_MLP.nii.gz')
