import math
import os
from typing import Dict, Tuple, Union

import functorch
import matplotlib.pyplot as plt
# import pytorch_lightning as pl
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import torchvision
from einops import rearrange

batch_size = 784
max_iter = 5
epochs = 5
epochs2 = 200
num_workers = os.cpu_count()
device = [0] if torch.cuda.is_available() else []

torch.random.manual_seed(0)


def exists(val):
    return val is not None


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# siren layer
class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


# siren network
class SirenNet(pl.LightningModule):
    def __init__(
        self,
        dim_in=3,
        dim_hidden=64,
        dim_out=1,
        num_layers=4,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.losses = []

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(
                Siren(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                )
            )

        final_activation = (
            nn.Identity() if not exists(final_activation) else final_activation
        )
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
        )

    def forward(self, x, mods=None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, "d -> () d")

        return self.last_layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)

        loss = F.mse_loss(z, y)
        self.losses.append(loss.detach().cpu().numpy())

        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer

    def set_parameters(self, theta):
        """
        Manually set parameters using matching theta, no foolproof
        """
        p_dict = self.state_dict()
        for p, thet in zip(p_dict, theta):
            p_dict[p] = thet.data
        self.load_state_dict(p_dict)


class Optimizer(nn.Module):
    """
    TODO: Linear instead of identity
    -Graidnet normalisation
    """

    def __init__(self, input_shape, hidden_size, num_layers=2, preproc=False) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=np.prod(input_shape),
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        # self.lstm = nn.LSTM(input_size=2 * input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, np.prod(input_shape))  #
        # self.output = nn.Identity()
        self.input_shape = input_shape
        # cell and hidden states are attributes of the class in this example
        self.register_buffer(
            "cell_state", torch.randn(num_layers, hidden_size), persistent=True
        )
        self.register_buffer(
            "hidden_state", torch.randn(num_layers, hidden_size), persistent=True
        )
        self.preproc = (
            preproc
        )  # WIP: preprocessing as discussed in the annex of "learning to learn by gradient descent by gradient descent"
        self.preproc_factor = 10.0
        self.preproc_threshold = np.exp(-self.preproc_factor)

    def reset_state(
        self,
    ):  # We reset LSTM states at each epoch, no need to carry it over between steps
        self.cell_state = torch.zeros_like(self.cell_state)
        self.hidden_state = torch.zeros_like(self.hidden_state)

    def forward(self, x):
        if self.preproc:
            # TODO: Not adapted to attribute structure
            # Implement preproc described in Appendix A

            # Note: we do all this work on tensors, which means
            # the gradients won't propagate through inp. This
            # should be ok because the algorithm involves
            # making sure that inp is already detached.
            inp = x.data
            inp2 = torch.zeros(inp.size(), 2, device=x.device)
            keep_grads = (torch.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (
                torch.log(torch.abs(inp[:, keep_grads]) + 1e-8) / self.preproc_factor
            ).squeeze()
            inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads]).squeeze()

            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (
                float(np.exp(self.preproc_factor)) * inp[~keep_grads]
            ).squeeze()
            x = inp2.requires_grad(True)
        # print(x.shape)
        # x = x.squeeze().unsqueeze(-1)
        # print(x.shape)
        x = x.detach()
        out, (new_cell_state, new_hidden_state) = self.lstm(
            x, (self.cell_state, self.hidden_state)
        )
        self.cell_state = new_cell_state.detach()
        self.hidden_state = new_hidden_state.detach()

        return self.output(out).reshape(self.input_shape)


mnist_dataset = torchvision.datasets.MNIST(
    root="/home/benjamin/Documents/Datasets", download=False
)
digit = mnist_dataset[25]  # a PIL image
digit_tensor = torchvision.transforms.ToTensor()(digit[0]).squeeze()
digit_tensor = digit_tensor * 2 - 1
digit_shape = digit_tensor.shape

x = torch.linspace(-1, 1, digit_shape[0])
y = torch.linspace(-1, 1, digit_shape[1])
mgrid = torch.stack(torch.meshgrid(x, y), dim=-1)

x_flat = torch.Tensor(mgrid.reshape(-1, 2))
y_flat = torch.Tensor(digit_tensor.flatten()).unsqueeze(-1)

dataset = torch.utils.data.TensorDataset(x_flat, y_flat)
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)

model = SirenNet(
    dim_in=2,
    dim_hidden=16,
    dim_out=1,
    num_layers=2,
    w0=1.0,
    w0_initial=30.0,
    use_bias=True,
    final_activation=None,
)

model_losses_adam_opt = []
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for _ in range(epochs2):
    x, y = next(iter(train_loader))
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"Loss: {loss.data}")
    model_losses_adam_opt.append(loss.detach().numpy())

# # Classical training test :: Test purpose
# trainer = pl.Trainer(gpus=device, max_epochs=epochs)
# trainer.fit(model, train_loader)

# test_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
# )

# pred = torch.cat(trainer.predict(model, test_loader))
# image = pred.reshape(digit_shape)
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(image.detach().numpy())
# axes[1].imshow(digit_tensor.detach().numpy())
# plt.show()

# TODO: reinstance of the model does not yield exact same training results even with fixed seed, maybe need to do it externally and reset parameters each time
model = SirenNet(
    dim_in=2,
    dim_hidden=16,
    dim_out=1,
    num_layers=2,
    w0=1.0,
    w0_initial=30.0,
    use_bias=True,
    final_activation=None,
)

model_func, model_params = functorch.make_functional(model)

optimizer_list = [
    Optimizer(input_shape=parameter.shape, hidden_size=np.prod(parameter.shape) * 10)
    for idx, parameter in enumerate(model_params)
]
opt = torch.optim.Adam([p for g in optimizer_list for p in g.parameters()], lr=1e-3)

log_outer_losses = []

for epoch in range(epochs):
    theta_zero = tuple(model_params)
    theta = list(theta_zero)
    log_model_losses = []
    for optimizer in optimizer_list:
        optimizer.reset_state()
    for it in range(max_iter):
        x, y = next(iter(train_loader))
        y_pred = model_func(theta, x)
        model_loss = F.mse_loss(y, y_pred)
        theta_gradients = torch.autograd.grad(
            model_loss, theta, create_graph=True, allow_unused=True
        )

        for i in range(len(theta)):
            theta_update = optimizer_list[i](
                theta_gradients[i]
                .reshape(1, np.prod(theta_gradients[i].shape))
                .detach()
            )
            theta_update = theta_update.reshape(theta[i].shape)
            theta[i] = theta[i] - theta_update

        if it == 0:
            outer_loss = model_loss
        else:
            outer_loss = (
                outer_loss + ((it + 1) / max_iter) ** 2 * model_loss
            )  # Tests show that weighted the later loss stronger result in better convergence. We indeed want to aim for later loss
            # outer_loss = outer_loss + model_loss
        log_model_losses.append(model_loss.detach().numpy())
        log_outer_losses.append(outer_loss.detach().numpy())

    opt.zero_grad()
    outer_loss.backward()
    opt.step()

    print(f"outer loss: {outer_loss.data}, model_loss: {model_loss.data}")

model = SirenNet(
    dim_in=2,
    dim_hidden=16,
    dim_out=1,
    num_layers=2,
    w0=1.0,
    w0_initial=30.0,
    use_bias=True,
    final_activation=None,
)
model.set_parameters(theta)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# replaced with fit ?
for _ in range(epochs2):
    x, y = next(iter(train_loader))
    y_pred = model(x)
    loss = F.mse_loss(y, y_pred)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"Loss: {loss.data}")
    log_model_losses.append(loss.detach().numpy())


# pred = model(x_flat)
# image = pred.reshape(digit_shape)
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(image.detach().numpy())
# axes[1].imshow(digit_tensor.detach().numpy())
# plt.show()

# Transfert learning ?
digit = mnist_dataset[5]  # a PIL image
digit_tensor = torchvision.transforms.ToTensor()(digit[0]).squeeze()
digit_tensor = digit_tensor * 2 - 1

x_flat = torch.Tensor(mgrid.reshape(-1, 2))
y_flat = torch.Tensor(digit_tensor.flatten()).unsqueeze(-1)
dataset = torch.utils.data.TensorDataset(x_flat, y_flat)
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)

model = SirenNet(
    dim_in=2,
    dim_hidden=16,
    dim_out=1,
    num_layers=2,
    w0=1.0,
    w0_initial=30.0,
    use_bias=True,
    final_activation=None,
)
model.set_parameters(theta)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

transfert_losses = []
for _ in range(epochs2):
    x, y = next(iter(train_loader))
    y_pred = model(x)
    loss = F.mse_loss(y, y_pred)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"Loss: {loss.data}")
    transfert_losses.append(loss.detach().numpy())


fig1 = plt.plot(range(epochs2), model_losses_adam_opt)
fig2 = plt.plot(range(max_iter + epochs2), log_model_losses)
fig3 = plt.plot(range(epochs2), transfert_losses)

plt.show()

class MAML:
    '''
    catch losses ?
    '''
    def __init__(self, model) -> None:
        self.optimizee = model
        self.optimizer = Optimizer(input_shape, hidden_size, num_layers=2, preproc=False)
        
    def optimize(self, steps):
        model_func, model_params = functorch.make_functional(model)

        optimizer_list = [
            Optimizer(input_shape=parameter.shape, hidden_size=np.prod(parameter.shape) * 10)
            for idx, parameter in enumerate(model_params)
        ]
        opt = torch.optim.Adam([p for g in optimizer_list for p in g.parameters()], lr=1e-3)

        log_outer_losses = []

        for step in range(steps):
            theta_zero = tuple(model_params)
            theta = list(theta_zero)
            log_model_losses = []
            for optimizer in optimizer_list:
                optimizer.reset_state()
            for it in range(max_iter):
                x, y = next(iter(train_loader))
                y_pred = model_func(theta, x)
                model_loss = F.mse_loss(y, y_pred)
                theta_gradients = torch.autograd.grad(
                    model_loss, theta, create_graph=True, allow_unused=True
                )

                for i in range(len(theta)):
                    theta_update = optimizer_list[i](
                        theta_gradients[i]
                        .reshape(1, np.prod(theta_gradients[i].shape))
                        .detach()
                    )
                    theta_update = theta_update.reshape(theta[i].shape)
                    theta[i] = theta[i] - theta_update

                if it == 0:
                    outer_loss = model_loss
                else:
                    outer_loss = (
                        outer_loss + ((it + 1) / max_iter) ** 2 * model_loss
                    )  # Tests show that weighted the later loss stronger result in better convergence. We indeed want to aim for later loss
                    # outer_loss = outer_loss + model_loss
                log_model_losses.append(model_loss.detach().numpy())
                log_outer_losses.append(outer_loss.detach().numpy())

            opt.zero_grad()
            outer_loss.backward()
            opt.step()
