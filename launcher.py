"""
Barebone laucnher for tests

TODO:
-workers and device in conf OR include directly into class. prob if too many workers
"""
# HYDRA_FULL_ERROR=1

import json
import os

import pytorch_lightning as pl
import torch

import hydra
from datamodules import MriDataModule
from hydra.utils import call, get_class, instantiate
from models import (ModulatedSiren,  # hashsiren to be done, modulated siren to be correced, probably recode models with config as arg
                    SirenNet)
from omegaconf import DictConfig, OmegaConf


class FakeOptimizer:
    def __init__(self, arg1, arg2, arg3):
        super.__init__()
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3


num_workers: int = os.cpu_count()
device = [0] if torch.cuda.is_available() else []

# either instanciate a dataclass config or if else?
@hydra.main(version_base=None, config_path="config", config_name="base")
def app(cfg: DictConfig):
    global config
    config = cfg

    # test fakeoptimizer
    opt = get_class(cfg.optimizer)
    # #datamodule
    # datamodule = get_class(config.datamodule.cls)(config=config.datamodule)
    # datamodule.prepare_data()
    # datamodule.setup()

    # train_loader = datamodule.train_dataloader()
    # test_loader = datamodule.test_dataloader()

    # model_cls = get_class(config.siren.cls)
    # model = model_cls(config=config.siren)
    # trainer = pl.Trainer(gpus=device, max_epochs=config.training.epochs, accumulate_grad_batches=config.training.accumulate_grad_batches)
    # trainer.fit(model, train_loader)


if __name__ == "__main__":
    app()
