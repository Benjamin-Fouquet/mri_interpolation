import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

# create a small regression exemple

# retake the temperature example, with noise

t_c = [10, 25, 43, 65, 47, 87, 34, 54, 23, 32, 19, 48, 92, 76]
t_u = [(i * 3.14 + 5) for i in t_c]

batch_size = 10
epochs = 10

fig = plt.plot(t_u, t_c, "o")
plt.show()


class MLP(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.learning_rate = learning_rate
        self.losses = []
        self.loss = F.mse_loss

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x)
        loss = self.loss(y, y_hat)
        self.losses.append(loss)
        if batch_idx % 20 == 0:
            # solve
            pass
        return loss

    def solve(self, batch):
        x_0, y_0 = batch
        y_hat = self.layers(x_0)
        optimizee_losses = []
        for i in range(self.opti_step):
            optimizee_loss = self.loss(y_0, y_hat)
            optimizee_losses.append(optimizee_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x)
        y_true = self.y_scaler.inverse_transform(y.cpu().numpy())
        y_pred = self.y_scaler.inverse_transform(y_hat.cpu().numpy())
        loss = self.loss(y_true, y_pred)
        return loss

    def validation_epoch_end(self, val_step_outputs):
        loss = sum(val_step_outputs) / len(val_step_outputs)
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            return self(x)

    def configure_optimizers(self):
        # here the magic ?
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


model = MLP()

t_u = torch.FloatTensor(t_u).unsqueeze(-1)
t_c = torch.FloatTensor(t_c).unsqueeze(-1)

dataset = torch.utils.data.TensorDataset(t_u, t_c)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

trainer = pl.Trainer(gpus=[], max_epochs=epochs)
trainer.fit(model, loader)

inputs = torch.randn(1000).unsqueeze(-1)
inputs = inputs * inputs * 100

test_dataset = torch.utils.data.TensorDataset(inputs, inputs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

pred = torch.concat(trainer.predict(model, test_loader))
fig = plt.plot(pred.detach().numpy(), pred.detach().numpy(), "o")
plt.show()


class Optimizee(pl.LightningModule):
    def __init__(self, optimizer, learning_rate=1e-3):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.learning_rate = learning_rate
        self.losses = []
        self.loss = F.mse_loss
        self.optimizer = optimizer

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        # conditional optimizer ?
        return self.optimizer


class Optimizer(pl.LightningModule):
    def __init__(self, optimizee_losses, learning_rate=1e-3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class Runner(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.optimizer = Optimizer(optimizee_losses=torch.ones(5))
        self.optimizee = Optimizee(optimizer=self.optimizer)
        self.optimizee_losses = torch.ones(5)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.optimizee(x)
        optimizee_loss = self.optimizee.loss(y, y_pred)
        # self.optimizee_losses = torch.cat((self.optimizee_losses, optimizee_loss.reshape(1)))
        if batch_idx == 0:
            self.optimizee_losses = optimizee_loss.reshape(1)
        else:
            self.optimizee_losses = torch.cat(
                (self.optimizee_losses, optimizee_loss.reshape(1))
            )
        self.optimizer.optimizee_losses = self.optimizee_losses

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self.optimizee(x)

    def configure_optimizers(self):
        pass


runner = Runner()
trainer = pl.Trainer(gpus=[], max_epochs=epochs)
trainer.fit(runner, loader)

inputs = torch.randn(1000).unsqueeze(-1)
inputs = inputs * inputs * 100

test_dataset = torch.utils.data.TensorDataset(inputs, inputs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

pred = torch.concat(trainer.predict(model, test_loader))
fig = plt.plot(pred.detach().numpy(), pred.detach().numpy(), "o")
plt.show()
