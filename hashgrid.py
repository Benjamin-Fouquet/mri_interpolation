'''
Tentative implementing the following paper in pure python: Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
Basically, hash encoding of features for implicit neural representation
TODO:
-decoder
-model encapsulating both
-test of forward and backward
'''
import torch
import numpy as np 
import pytorch_lightning as pl 
from typing import Any
import os
import nibabel as nib

class HashEncoding(pl.LightningModule):
    '''
    Pure python version of tcnn encoding class, for educational purposes. Same basic functionality, without the optimization (pure python indeed!)
    '''
    def __init__(self, dim_in, dim_out, n_levels, n_features_per_level, log2_hashmap_size, base_resolution, per_level_scale, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size #needed ? maybe you need to have the keys fixed so that you can create the parameter space and optimize
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.hashtable = torch.nn.parameter.Parameter(torch.zeros((self.n_levels, 2 ** self.log2_hashmap_size, self.n_features_per_level), dtype=torch.float16)) #stable nn.parameters with indices access ? probably
        self.primes = (1, 2_654_435_761, 805_459_861, 73_856_093)
        self.lr = 5e-3
        
    def forward(self, x):
        '''
        Only first level for hte moment
        '''
        #ugly unoptimized loop
        z = []
        for coords in x:
            #calculate the hash value
            hash_value = int(coords[0]) ^ int(coords[1]) * self.primes[1] ^ int(coords[2]) * self.primes[2] % 2 ** self.log2_hashmap_size
            z.append(self.hashtable[0][hash_value])
        return torch.FloatTensor(z)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer
    
class Decoder(pl.LightningModule):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers, activation,*args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dim_in = dim_in
        self.dim_out = dim_out 
        self.dim_hidden = dim_hidden 
        self.n_layers = n_layers 
        self.activation = getattr(torch.nn, activation)
        self.lr = 5e-3
        layers = []
        for i in range(n_layers):
            layers.append(torch.nn.Linear(in_features=self.dim_in if i == 0 else self.dim_hidden, out_features=dim_out if i == n_layers - 1 else self.dim_hidden))
            layers.append(self.activation())
        
        self.layers = torch.nn.Sequential(*layers)
         
    def forward(self, x):
        return self.layers(x)  
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer
        
#next step, create the pytorch training loop with the two things and try to backpropagate, see if anything is learned
num_epochs = 20
batch_size = 4000
num_workers = os.cpu_count()
image_file = 'data/HCP/100307_T2.nii.gz'

#Read first image
image = nib.load(image_file)
data = image.get_fdata()

#Create grid
dim = 3
nx = data.shape[0]
ny = data.shape[1]
nz = data.shape[2]
nmax = np.max([nx,ny,nz])

x = torch.linspace(0, 1, steps=nx)
y = torch.linspace(0, 1, steps=ny)
z = torch.linspace(0, 1, steps=nz)

mgrid = torch.stack(torch.meshgrid(x,y,z,indexing='ij'), dim=-1)

#Convert to X=(x,y,z) and Y=intensity
X = torch.Tensor(mgrid.reshape(-1,dim))
Y = torch.Tensor(data.flatten())

#Normalize intensities between [-1,1]
Y = (Y - torch.min(Y)) / (torch.max(Y) - torch.min(Y)) * 2 - 1
Y = torch.reshape(Y, (-1,1))

dataset = torch.utils.data.TensorDataset(X,Y)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

model = torch.nn.Sequential(HashEncoding(dim_in=3, dim_out=1, n_levels=8, n_features_per_level=2, log2_hashmap_size=16, base_resolution=32, per_level_scale=2), Decoder(dim_in=1, dim_hidden=64, dim_out=1, n_layers=3, activation='ReLU'))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(num_epochs):
    # TRAINING LOOP
    for train_batch in loader:
        x, y = train_batch
        
        y_pred = model(x)

        loss = F.mse_loss(y_pred, y)
        print(f"epoch: {epoch}")
        print("train loss: ", loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
        