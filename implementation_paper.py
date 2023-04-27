'''
Implementation 'Continuous Longitudinal Fetus Brain Atlas Construction via Implicit Neural Representation' using tinycuda
'''
import tinycudann as tcnn 
import torch
import pytorch_lightning as pl 
import torch.nn.functional as F
import json
import nibabel as nib 
from dataclasses import dataclass
import os
from types import MappingProxyType
import numpy as np

@dataclass
class BaseConfig:
    checkpoint_path = None #'lightning_logs/version_384/checkpoints/epoch=99-step=100.ckpt'
    image_path: str = '/mnt/Data/FetalAtlas/template_T2.nii.gz'
    image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_shape = nib.load(image_path).shape
    batch_size: int = 1000000 #~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 50
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None #MappingProxyType({200: 2}) #MappingProxyType({0: 5})

    hashconfig_path: str = 'config/hash_config.json'

    # Network parameters
    dim_in: int = len(image_shape)
    dim_hidden: int = 256 #should match n_frequencies
    dim_out: int = 1
    num_layers: int = 5
    skip_connections = (5, 11,)
    w0: float = 30.0
    w0_initial: float = 30.0
    use_bias: bool = True
    final_activation = None
    lr: float = 1e-4  # G requires training with a custom lr, usually lr * 0.1 
    # datamodule: pl.LightningDataModule = MriFramesDataModule
    # model_cls: pl.LightningModule = MultiHashMLP  
    n_frames: int = 15
    # n_frames: int = image_shape[-1] if len(image_shape) == 4 else None

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")

#freq encoding tiny cuda
class FreqMLP(pl.LightningModule):
    '''
    Lightning module for HashMLP. 
    '''
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        n_layers,
        skip_connections,
        config,
        lr,
        *args,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.skip_connections = skip_connections #index of skip connections, starting from 0
        self.lr = lr
        self.latents = []

        self.encoder = tcnn.Encoding(n_input_dims=dim_in, encoding_config=config['encoding'])
        # self.decoder = tcnn.Network(n_input_dims=self.encoder.n_output_dims, n_output_dims=dim_out, network_config=config['network'])
        self.decoder = torch.nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                in_features = self.dim_in * 2 * config['encoding']['n_frequencies']
            elif i in self.skip_connections:
                in_features = self.dim_in * 2 * config['encoding']['n_frequencies'] + self.dim_hidden
            else:
                in_features = self.dim_hidden
            block = torch.nn.Sequential(
                torch.nn.Linear(in_features=in_features, out_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden),
                torch.nn.BatchNorm1d(num_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden), #you can do batochnorm 3D + 1D and cat after
                torch.nn.ReLU()
            )
            self.decoder.append(block)
            

    def forward(self, x):
        x = self.encoder(x)
        skip = x.clone()
        for idx, layer in enumerate(self.decoder):
            if idx in self.skip_connections:
                x = torch.cat(skip, x)
            x = layer(x)
        return x 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5) #weight_decay=1e-5
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)

        loss = F.mse_loss(y_pred, y)

        self.log("train_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        return y_pred

    def get_latents(self):
        return self.latents

with open('config/freq_config.json') as f:
    enco_config = json.load(f)
    
config = BaseConfig()
model = FreqMLP(dim_in=config.dim_in, 
                dim_hidden=config.dim_hidden, 
                dim_out=config.dim_out, 
                n_layers=config.num_layers, 
                skip_connections=config.skip_connections,
                config=enco_config,
                lr=config.lr)

#include batch norm, make a mock dataset (why mock, you can do a normal one)
mri_image = nib.load(config.image_path)

Y = torch.FloatTensor(mri_image.get_fdata(dtype=np.float32)).flatten().unsqueeze(-1)

axes = []
for s in mri_image.shape:
    axes.append(torch.linspace(0, 1, s))

mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

coords = torch.FloatTensor(mgrid)
X = coords.reshape(len(Y), config.dim_in)

dataset = torch.utils.data.TensorDataset(X, Y)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=os.cpu_count() // 4)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=os.cpu_count() // 4)  
                
trainer = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
    precision=16,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)
# trainer = pl.Trainer(gpus=config.device, max_epochs=config.epochs)
trainer.fit(model, train_loader)

#create a prediction
pred = torch.concat(trainer.predict(model, test_loader))
            
im = pred.reshape(mri_image.shape)
im = im.detach().cpu().numpy()
im = np.array(im, dtype=np.float32)
nib.save(nib.Nifti1Image(im, affine=np.eye(4)), 'pred.nii.gz')               

