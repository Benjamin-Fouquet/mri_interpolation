from typing import List, Optional, Union
from pytorch_lightning.utilities.types import LRSchedulerTypeUnion
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
import math
import rff
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.nn
from functools import lru_cache
import torch.utils.data
import matplotlib.pyplot as plt


for idx, row in enumerate(im):
    im[idx] = np.sin(idx)

plt.imshow(im)
plt.
plt.savefig('out.png')

axes = []
for s in im.shape:
    axes.append(torch.linspace(0, 1, s))
    
mgrid = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)