'''
For H encoding experimentation and interpolation, derived from code implementatation

Implementation 'Continuous Longitudinal Fetus Brain Atlas Construction via Implicit Neural Representation' using tinycuda

current state: Test if the differnetial encoding and reordering of T is beneficial for resutlts. Maybe try first 1 encoder but with reordered T, then dual encoder
'''
from typing import List, Optional, Union
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
# import tinycudann as tcnn 
import torch
import pytorch_lightning as pl 
import torch.nn.functional as F
import nibabel as nib 
from dataclasses import dataclass
import os
from types import MappingProxyType
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from math import pi
import torch.utils.data
import matplotlib.pyplot as plt
import encoding
from skimage import metrics
import tinycudann as tcnn

torch.manual_seed(1337)

@dataclass
class BaseConfig:
    checkpoint_path = None #'lightning_logs/version_61/checkpoints/epoch=199-step=12000.ckpt'
    # image_path: str = '/mnt/Data/FetalAtlas/template_T2.nii.gz'
    image_path: str = '/mnt/Data/Equinus_BIDS_dataset/sourcedata/sub_E01/sub_E01_dynamic_MovieClear_active_run_12.nii.gz'
    image_shape = nib.load(image_path).shape
    interp_shape = (352, 352, 30)
    batch_size: int = 20000 #~max #int(np.prod(image_shape)) #int(np.prod(image_shape)) if len(image_shape) < 4 else 1 #743424 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 1000
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None 
    encoder_type: str = 'hash' #   
    # Network parameters
    n_levels: int = 8
    n_features_per_level: int = 4
    log2_hashmap_size: int = 23
    base_resolution: MappingProxyType = (64, 64, 5)
    finest_resolution: MappingProxyType = (352, 352, 15)
    # base_resolution: int = 64
    # finest_resolution: int = 512
    per_level_scale: int = 1.2
    interpolation: str = "Linear" #can be "Nearest", "Linear" or "Smoothstep", not used if not 'tcnn' encoder_type
    dim_in: int = len(image_shape)
    dim_hidden: int = 64 
    dim_out: int = 1
    num_layers: int = 2
    lr: float = 5e-3  # G requires training with a custom lr, usually lr * 0.1 
    dropout: float = 0.0

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch size", type=int, required=False)
    parser.add_argument("--epochs", help="Number of epochs", type=int, required=False)
    parser.add_argument("--image_path", help="path of image", type=str, required=False)
    parser.add_argument("--encoder_type", help="tcnn or classic", type=str, required=False)
    parser.add_argument("--n_frequencies", help="number of encoding frequencies", type=int, required=False)
    parser.add_argument("--n_frequencies_t", help="number of encoding frequencies for time", type=int, required=False)
    parser.add_argument("--lr", help="learning rate", type=float, required=False)
    parser.add_argument("--dim_hidden", help="hidden dimension for decoder", type=int, required=False)
    parser.add_argument("--n_levels", help="number of levels", type=int, required=False)
    parser.add_argument("--n_features_per_level", help="n_features_per_level", type=int, required=False)
    parser.add_argument("--num_layers", help="number of layers for decoder", type=int, required=False)
    parser.add_argument("--dropout", help="Dropout for decoder", type=float, required=False)

    

    args = parser.parse_args()

def export_to_txt(dict: dict, file_path: str = "") -> None:
    '''
    Helper function to export dictionary to text file
    '''
    with open(file_path + "config.txt", "a+") as f:
        for key in dict:
            f.write(str(key) + " : " + str(dict[key]) + "\n")

config = BaseConfig()

# parsed argument -> config
for key in args.__dict__:
    if args.__dict__[key] is not None:
        config.__dict__[key] = args.__dict__[key]


class HashMLP(pl.LightningModule):
    '''
    Lightning module for HashMLP. 
    '''
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        n_layers,
        encoder_type,
        n_levels,
        n_features_per_level,
        log2_hashmap_size,
        base_resolution,
        finest_resolution,
        per_level_scale,
        interpolation,
        lr,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.encoder_type = encoder_type
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.per_level_scale = per_level_scale
        self.interpolation = interpolation
        self.lr = lr

        if self.encoder_type == 'tcnn':
            self.encoder = torch.nn.utils.parametrizations.spectral_norm(tcnn.Encoding(n_input_dims=(self.dim_in), encoding_config= {"otype": "HashGrid", "n_levels": self.n_levels, "n_features_per_level": self.n_features_per_level, "log2_hashmap_size": self.log2_hashmap_size, "base_resolution": self.base_resolution, "per_level_scale": self.per_level_scale, "interpolation": self.interpolation}, dtype=torch.float32), name='params', n_power_iterations=4, eps=1e-12, dim=None)
        
        else: 
            if isinstance(self.base_resolution, int):
                self.encoder = encoding.MultiResHashGrid(
                    dim=self.dim_in, 
                    n_levels=self.n_levels, 
                    n_features_per_level=self.n_features_per_level,
                    log2_hashmap_size=self.log2_hashmap_size,
                    base_resolution=self.base_resolution,
                    finest_resolution=self.finest_resolution,
                    )
            else:
                self.encoder = encoding.MultiResHashGridV2(
                    dim=self.dim_in, 
                    n_levels=self.n_levels, 
                    n_features_per_level=self.n_features_per_level,
                    log2_hashmap_size=self.log2_hashmap_size,
                    base_resolution=self.base_resolution,
                    finest_resolution=self.finest_resolution,
                    )
        
        self.encoding_dim_out = self.n_levels * self.n_features_per_level

        self.decoder = torch.nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                in_features = self.encoding_dim_out
            else:
                in_features = self.dim_hidden
            block = torch.nn.Sequential(
                torch.nn.utils.parametrizations.spectral_norm(torch.nn.Linear(in_features=in_features, out_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden), n_power_iterations=4, eps=1e-12, dim=None),
                # torch.nn.Linear(in_features=in_features, out_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden),
                torch.nn.BatchNorm1d(num_features=self.dim_out if i == (self.n_layers - 1) else self.dim_hidden), #you can do batchnorm 3D + 1D and cat after
                # torch.nn.ReLU()
                torch.nn.GELU(),
                torch.nn.Dropout(p=config.dropout, inplace=False),
            )
            self.decoder.append(block)
            

    def forward(self, x):
        x = self.encoder(x)
        for idx, layer in enumerate(self.decoder):
            x = layer(x)
        return x

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5) #weight_decay=1e-5
        return self.optimizer

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
    
    def lr_schedulers(self) -> LRSchedulerTypeUnion | List[LRSchedulerTypeUnion] | None:
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=10, verbose=True)
        return self.scheduler
    
    def on_train_end(self) -> None:
        writer = SummaryWriter(log_dir=self.logger.log_dir)
        writer.add_text(text_string=str(config), tag='configuration')
        writer.close()
          
mri_image = nib.load(config.image_path)

data = mri_image.get_fdata(dtype=np.float32)
data = data[:,:,3,:] #optional line for 2D + t experiments, speedup
config.image_shape = data.shape
config.dim_in = len(data.shape)

#interpolation tests
# data = data[..., ::2]

model = HashMLP(dim_in=config.dim_in, 
                dim_hidden=config.dim_hidden, 
                dim_out=config.dim_out, 
                n_layers=config.num_layers,
                encoder_type=config.encoder_type,
                n_levels=config.n_levels,
                n_features_per_level=config.n_features_per_level,
                log2_hashmap_size=config.log2_hashmap_size,
                base_resolution=config.base_resolution,
                finest_resolution=config.finest_resolution,
                per_level_scale=config.per_level_scale,
                interpolation=config.interpolation,
                lr=config.lr)

Y = torch.FloatTensor(data).reshape(-1, 1)
Y = Y / Y.max()

axes = []
for s in config.image_shape:
    axes.append(torch.linspace(0, 1, s))

mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

coords = torch.FloatTensor(mgrid)

if data.shape[-1] < 15: #conditional step if interp on even frames
    if config.dim_in ==3:
        coords = coords[:,:,::2,:]
    if config.dim_in ==4:
        coords = coords[:,:,:,::2,:]
X = coords.reshape(len(Y), config.dim_in)

dataset = torch.utils.data.TensorDataset(X, Y)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

#Tinyset, 1 timepoint. TODO: try with 2 later
tinyY= Y = torch.FloatTensor(data[..., [3, 7, 13]]).reshape(-1, 1) #TODO: Activate this for brain
# tinyY= Y = torch.FloatTensor(data[..., 4]).reshape(-1, 1)
tinyY = tinyY / Y.max()   

if len(config.image_shape) == 4:
    coords = torch.FloatTensor(mgrid[:,:,:,[3, 7, 13],:])
if len(config.image_shape) == 3:
    coords = torch.FloatTensor(mgrid[:,:,[3, 7, 13],:])

# if config.dim_in ==3:
#     coords = torch.FloatTensor(mgrid[:,:,4,:])d
# if config.dim_in ==4:
#     coords = torch.FloatTensor(mgrid[:,:,:,4,:])

tinyX = coords.reshape(len(tinyY), config.dim_in)
    
tinydataset = dataset = torch.utils.data.TensorDataset(tinyX, tinyY)
tiny_loader = torch.utils.data.DataLoader(tinydataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
   
trainer = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
    precision=32,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)

#pretraining
trainer.fit(model, tiny_loader)
# trainer.fit(model, train_loader)

#freeze decoder
for layer in model.decoder:
    layer.requires_grad = False
  
trainer = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
    precision=32,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)
trainer.fit(model, train_loader)

filepath = model.logger.log_dir + '/'
config.log = str(model.logger.version)

#create a prediction
pred = torch.concat(trainer.predict(model, test_loader))
            
# im = pred.reshape(config.image_shape)
im = pred.reshape(data.shape)
im = im.detach().cpu().numpy()
im = np.array(im, dtype=np.float32)
if len(im.shape) == 2:
    plt.imshow(im.T)
    plt.savefig(filepath + 'pred.png')
else:
    nib.save(nib.Nifti1Image(im, affine=np.eye(4)), filepath + 'pred.nii.gz')

interp_shapes = [(352, 352, 29), (352, 352, 59)]
# interp_shapes = [(117, 159, 126, 30)]
#ugly loop as placeholder
for shape in interp_shapes:    
    #dense grid
    config.interp_shape = shape
    Y_interp = torch.zeros((np.prod(config.interp_shape), 1))

    axes = []
    for s in config.interp_shape:
        axes.append(torch.linspace(0, 1, s))
            
    mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

    coords = torch.FloatTensor(mgrid)
    X_interp = coords.reshape(len(Y_interp), config.dim_in)    

    interp_dataset = torch.utils.data.TensorDataset(X_interp, Y_interp)
    interp_loader = torch.utils.data.DataLoader(interp_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    #create an interpolation
    interp = torch.concat(trainer.predict(model, interp_loader))

    interp_im = interp.reshape(config.interp_shape)
        
    interp_im = interp_im.detach().cpu().numpy()
    interp_im = np.array(interp_im, dtype=np.float32)
    nib.save(nib.Nifti1Image(interp_im, affine=np.eye(4)), filepath + f'interpolation{shape}.nii.gz')

config.export_to_txt(file_path=filepath)

output = im
ground_truth = nib.load(config.image_path).get_fdata()
ground_truth = ground_truth / ground_truth.max()

# # metrics
# with open(filepath + "scores.txt", "w") as f:
#     f.write("MSE : " + str(metrics.mean_squared_error(ground_truth, output)) + "\n")
#     f.write("PSNR : " + str(metrics.peak_signal_noise_ratio(ground_truth, output)) + "\n")
#     # if config.dim_in < 4:
#     #     f.write("SSMI : " + str(metrics.structural_similarity(ground_truth, output)) + "\n")
#     # f.write(
#     #     "training time  : " + str(training_stop - training_start) + " seconds" + "\n"
#     # )
#     f.write(
#         "Number of trainable parameters : "
#         + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
#         + "\n"
#     )  # remove condition if you want total parameters
#     f.write("Max memory allocated : " + str(torch.cuda.max_memory_allocated()) + "\n")

# plt.imshow(im)
# plt.savefig(filepath + 'out.png')

# #extract latents
# model.to('cuda')
# lats = model.encoder(X_interp.to('cuda'))

# os.mkdir(filepath + 'latents/')

# for idx, lat in enumerate(lats.T):
#     lat = lat.detach().cpu().numpy()
#     lat = lat.reshape((mri_image.shape[0], mri_image.shape[1], mri_image.shape[2], mri_image.shape[3] * config.interp_factor)) #only 2D
#     nib.save(nib.Nifti1Image(lat, affine=np.eye(4)), filepath + 'latents/' + f'{idx}.nii.gz')
   
# for i in range(4):
#     array = params[262144 * i: 262144 * (i + 1)].detach().cpu().numpy()
#     array = array.reshape(64, 64, 64)
#     nib.save(nib.Nifti1Image(array, affine=np.eye(4)), filepath + f'latents/paramap{i}.nii.gz')
    
# #create function for parameter linear interpolation when needed (ideally you could do it 1D)
# def regularize_hash(hash_table):
#     '''
#     Trial function for specific 64 ** 3 case
#     '''
#     array_list = []
#     assert len(hash_table) == 1048576
#     for i in range(4):
#         array = hash_table[262144 * i: 262144 * (i + 1)].detach().cpu().numpy()
#         array = array.reshape(64, 64, 64)
#         array_list.append(array) #append one array per level and feature
        
#     for array in array_list:
#         for idx, a in enumerate(array):
#             #if sum a too low
#             if a.sum() < 1e-3:
#                 #search neighbour valid slices
                        
#             #sum it up
#             #interpolate with neighbouring slices that have a value
#             #replace

# array = model.encoder.params.detach().cpu().numpy()

# array = array.reshape(config.base_resolution, config.base_resolution, config.base_resolution) #only works for some exemples

# #In test, does not work yet
# for idx, slc in enumerate(array):
#     if slc.sum() < 1e-2 and idx > 0:
#         array[idx] = array[(idx - 1)]
        
# array2 = model.encoder.params.detach().cpu().numpy()

'''
Source of smoothing: https://danielmuellerkomorowska.com/2020/06/02/smoothing-data-by-rolling-average-with-numpy/
Kernel smoothing 1D does not work, try 3D (mutliply anchor slices first if needed)
'''
# kernel_size = 50
# kernel = np.ones(kernel_size) / kernel_size
# array_convolved = np.convolve(array, kernel, mode='same')

##griddata on array, too time consuming
# axes = []
# # for s in array.shape:
# for s in (32, 32, 32):
#     axes.append(np.linspace(0, 1, s))
    
# mgrid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)

# grid = griddata(mgrid[::4].reshape(-1, 3), array[::4].reshape(-1, 1), xi=(mgrid))

# #set parameters
# p_dict = model.encoder.state_dict()
# p_dict['parametrizations.params.original'] = torch.FloatTensor(array_convolved)
    
# model.encoder.load_state_dict(p_dict)

# trainer = pl.Trainer(
#     gpus=config.device,
#     max_epochs=config.epochs,
#     accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
#     precision=32,
#     # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
# )
# trainer.fit(model, train_loader)
    
