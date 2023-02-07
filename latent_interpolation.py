'''
Latent space interpolation using siren

TODO:
latents are numpy in the loader, you will need to convert them to tensors (torch.HalfTensor works wonder)
'''
from models import SirenNet
import pytorch_lightning as pl
import os
import torch
from types import MappingProxyType
import nibabel
from dataclasses import dataclass
import glob
from torch.utils.data import Dataset, DataLoader
import tinycudann as tcnn
import json
import nibabel as nib
import numpy as np


class LatentDataset(Dataset):
    def __init__(self, latents_list, *args, **kwargs):
        super().__init__()

        assert len(latents_list) > 0, 'Latents_list is empty'
        self.latents = [torch.HalfTensor(lat) for lat in latents_list]
        self.times = torch.linspace(-1, 1, len(latents_list))

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.times[idx], self.latents[idx]

@dataclass
class SirenConfig:
    checkpoint_path = None
    batch_size: int = 1 # 28 * 28  #21023600 for 3D mri #80860 for 2D mri#784 for MNIST #2500 for GPU mem ?
    epochs: int = 1
    num_workers: int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    accumulate_grad_batches: MappingProxyType = None #MappingProxyType({200: 2}) #MappingProxyType({0: 5})

    # Network parameters
    dim_in: int = 1
    dim_hidden: int = 256
    dim_out: int = 64
    num_layers: int = 5
    n_sample: int = 3
    w0: float = 30.0
    w0_initial: float = 30.0
    use_bias: bool = True
    final_activation = None
    lr: float = 1e-3  # G requires training with a custom lr, usually lr * 0.1

    def export_to_txt(self, file_path: str = "") -> None:
        with open(file_path + "config.txt", "w") as f:
            for key in self.__dict__:
                f.write(str(key) + " : " + str(self.__dict__[key]) + "\n")

config = SirenConfig()

#build latent list
lats = []
lats_path = glob.glob('results/multi_hash_500/lat*', recursive=True)
for path in lats_path:
    lats.append(torch.load(path))

dataset = LatentDataset(lats)
train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count())
#create loader

#train siren
model = SirenNet(
        dim_in=config.dim_in,
        dim_hidden=config.dim_hidden,
        dim_out=config.dim_out,
        num_layers=config.num_layers,
        w0=config.w0,
        w0_initial=config.w0_initial,
        use_bias=config.use_bias,
        final_activation=config.final_activation,
        lr=config.lr,
    )

trainer = pl.Trainer(
    gpus=config.device,
    max_epochs=config.epochs,
    accumulate_grad_batches=dict(config.accumulate_grad_batches) if config.accumulate_grad_batches else None,
    precision=16,
    # callbacks=[pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)]
)
# trainer = pl.Trainer(gpus=config.device, max_epochs=config.epochs)
trainer.fit(model, train_loader)

with open("config/hash_config.json") as f:
    enco_config = json.load(f)

decoder = tcnn.Network(n_input_dims=enco_config["encoding"]["n_levels"] * enco_config["encoding"]["n_features_per_level"], n_output_dims=1, network_config=enco_config['network'])
decoder.load_state_dict(torch.load('results/multi_hash_500/decoder_statedict.pt'))

t_interp = torch.linspace(-1, 1, 60)

for t in t_interp:
    lat = model(t.reshape(1))
    pred = decoder(lat.to('cuda'))

pred = decoder(lat).reshape(352, 352, 6).detach().cpu().numpy()
pred = np.array(pred, dtype=np.float32)
nib.save(nib.Nifti1Image(pred, affine=np.eye(4)), 'interp_latent.nii.gz')

im0 = nib.load('im0.nii.gz')
im1 = nib.load('im2.nii.gz')

im0 = im0.get_fdata()
im1 = im1.get_fdata()

interp = 0.5 * im0 + 0.5 * im1
interp = interp /np.max(interp) * 2 - 1
nib.save(nib.Nifti1Image(interp, affine=np.eye(4)), 'interp_image.nii.gz')