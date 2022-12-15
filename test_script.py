import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule
import os
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import logging

import optuna
import pytorch_lightning as pl
from datamodules import MNISTDataModule
# from config.base import MNISTConfig
import os
from models import SirenNet, HashMLP, ModulatedSirenNet
import ray.tune as tune
from dataclasses import dataclass, field
from types import MappingProxyType
from config.base import BaseConfig
import json
from datamodules import MriDataModule
import optuna
import sys
from models import HashSirenNet

config = BaseConfig()

with open(BaseConfig.hashconfig_path) as f:
    enco_config = json.load(f)

dm = MriDataModule(config=config)
dm.prepare_data()
dm.setup()
train_loader = dm.train_dataloader()

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "example-study"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)

def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 6)
    dim_hidden = trial.suggest_int("dim_hidden", 64, 256, 64)
    w0 = trial.suggest_float("w0", 1.0, 70.0)
    w0_initial = trial.suggest_float("w0", 1.0, 70.0)
    # model_cls = trial.suggest_categorical("model_class", [SirenNet, ModulatedSirenNet])

    losses = []
     
    model = config.model_cls(
        dim_in=config.dim_in,
        dim_hidden=dim_hidden,
        dim_out=config.dim_out,
        num_layers=n_layers,
        w0=w0,
        w0_initial=w0_initial,
        use_bias=config.use_bias,
        final_activation=config.final_activation,
        lr=config.lr,
        config=enco_config,
        # coordinates_spacing=config.coordinates_spacing,
        # n_sample=config.n_sample
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    model.to('cuda')

    for epoch in range(config.epochs):
        # TRAINING LOOP
        for train_batch in train_loader:
            x, y = train_batch
            x = x.to("cuda")
            y = y.to("cuda")

            # x to hash
            y_pred = model(x)

            # x and mod to siren

            loss = F.mse_loss(y_pred, y)
            print(f"epoch: {epoch}")
            print("train loss: ", loss.item())
            losses.append(loss.detach().cpu().numpy())

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    return losses[-1]
                
study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=10000)
