'''
Not GPU compatible yet
TODO
Done:
-Check if parameter transfert is really taking place // works
-theta is correctly updated at iterations
-Correct output so that you have all losses correctly displayed Done
-LSTM parameters kept between iteration, but a reset on the conv still allow to learn somewhat, how ?! (ftm, reset_state does nothing for ConvOptimizer). To test later
'''

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Union, Dict

# import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import functorch
from torch.autograd import Variable

import math
from einops import rearrange
import pytorch_lightning as pl
import torchvision
import os
from dataclasses import dataclass, field
import sys
import argparse
import copy

@dataclass
class Config:
    batch_size: int = 784 #28 * 28
    inner_loop_it: int = 5
    outer_loop_it: int = 10
    epochs: int = 100
    num_workers:int = os.cpu_count()
    device = [0] if torch.cuda.is_available() else []
    fixed_seed: bool = True
    dataset_path: str = '/home/benjamin/Documents/Datasets'
    train_target = [1]
    test_target = [7]

    #Network parameters
    dim_in: int = 2
    dim_hidden: int = 32
    dim_out:int = 1
    num_layers:int = 2
    w0: float = 30.0
    w0_initial:float = 30.0
    use_bias: bool = True
    final_activation=None
    lr: float = 1e-3 #G requires training with a custom lr, usually lr * 0.1
    opt_type: str = 'LSTM'
    conv_channels = [8, 8, 8]

    comment: str = 'Mean initialisation'

    #output
    output_path:str = 'results_siren/'
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path)
    experiment_number:int = 0 if len(os.listdir(output_path)) == 0 else len(os.listdir(output_path))

    def export_to_txt(self, file_path: str = '') -> None:
        with open(file_path + 'config.txt', 'w') as f:
            for key in self.__dict__:
                f.write(str(key) + ' : ' + str(self.__dict__[key]) + '\n')

config = Config()

#name of arguments must match name of Config class /// TODO: compatibility with list target ?
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inner_loop_it', help='Inner loop iterations', type=int, required=False)
    parser.add_argument('--outer_loop_it', help='Outer loop iterations', type=int, required=False)
    parser.add_argument('--epochs', help='Number of epochs', type=int, required=False)
    parser.add_argument('--train_target', help='train digit', type=int, required=False)
    parser.add_argument('--test_target', help='test digit', type=int, required=False)
    parser.add_argument('--opt_type', help='optimizer model type', type=str, required=False)

    args = parser.parse_args()

#parsed argument -> config
for key in args.__dict__:
    if args.__dict__[key] is not None:
        config.__dict__[key] = args.__dict__[key]

#Correct ouput_path
filepath = config.output_path + str(config.experiment_number) + '/'
if os.path.isdir(filepath) is False:
    os.mkdir(filepath)

if config.fixed_seed:
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
        '''
        Manually set parameters using matching theta, not foolproof
        '''
        p_dict = self.state_dict()
        for p, thet in zip(p_dict, theta):
            p_dict[p] = thet.data
        self.load_state_dict(p_dict)
        self.eval() #supposed to be important when you set parameters or load state

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
        self.output = nn.Linear(hidden_size, np.prod(input_shape)) #
        # self.output = nn.Identity()
        self.input_shape = input_shape
        # cell and hidden states are attributes of the class in this example
        self.register_buffer(
            "cell_state", torch.randn(num_layers, hidden_size), persistent=True
        )
        self.register_buffer(
            "hidden_state", torch.randn(num_layers, hidden_size), persistent=True
        )
        self.preproc = preproc  # WIP: preprocessing as discussed in the annex of "learning to learn by gradient descent by gradient descent"
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

class ConvOptimizer(nn.Module):

    '''
    Marche que pour couche du centre, only 2D si tu fais une conv par couche, essayer full stack parameters ?
    '''
    def __init__(self, input, channels=[32, 32, 32], activation_func=None)->None:
        super().__init__()
        #Build the layer system
        conv_layer = nn.Conv2d
        layers = []
        for idx in range(len(channels)):
            in_channels = channels[idx - 1] if idx > 0 else 1
            out_channels = channels[idx]
            layer = conv_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

            layers.append(layer)
            if activation_func:
                layers.append(activation_func)

        last_layer = conv_layer(
            in_channels=channels[-1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        layers.append(last_layer)
        self.model = nn.Sequential(*layers)

    def reset_state(
        self,
    ):  
        pass
        # for parameter in self.parameters():
        #     parameter.data = torch.randn(parameter.shape) * 0.01
        # #TODO: Better initialisation using noraml distribution ? See: https://pytorch.org/docs/stable/nn.init.html
        return None

    def forward(self, x):
        return self.model(x)

#####################
#DATASET PREPARATION#
#####################
mnist_dataset = torchvision.datasets.MNIST(
    root=config.dataset_path, download=False
)

# #get all digits from train_targets and mean them
# mean_digit_tensor = torch.zeros(28, 28)
# for digit_idx, target in enumerate(mnist_dataset.targets):
#     if int(target) in config.train_target:
#         mean_digit_tensor += torchvision.transforms.ToTensor()(mnist_dataset[digit_idx][0]).squeeze()
        
# #normalization
# mean_digit_tensor = mean_digit_tensor / torch.max(mean_digit_tensor)
# mean_digit_tensor = mean_digit_tensor * 2 - 1

#fetch the wanted train digit, 1 digit version
for digit_idx, target in enumerate(mnist_dataset.targets):
    if int(target) in config.train_target:
        digit = mnist_dataset[digit_idx][0] # a PIL image
        break

digit_tensor = torchvision.transforms.ToTensor()(digit).squeeze()

#normalization
digit_tensor = digit_tensor * 2 - 1
digit_shape = digit_tensor.shape

x = torch.linspace(-1, 1, digit_shape[0])
y = torch.linspace(-1, 1, digit_shape[1])
mgrid = torch.stack(torch.meshgrid(x, y), dim=-1)

x_flat = torch.Tensor(mgrid.reshape(-1, 2))
y_flat = torch.Tensor(digit_tensor.flatten()).unsqueeze(-1)
# mean_y_flat = torch.Tensor(mean_digit_tensor.flatten()).unsqueeze(-1)

#This dataset contains 1 target digit
dataset = torch.utils.data.TensorDataset(x_flat, y_flat)
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
)

# #This dataset contains the mean of ALL target datasets
# mean_dataset = torch.utils.data.TensorDataset(x_flat, mean_y_flat)
# mean_train_loader = torch.utils.data.DataLoader(
#     mean_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
# )

# #This dataset returns a different digit from the target class each time
# targets_list = []
# for digit_idx, target in enumerate(mnist_dataset.targets):
#     if int(target) in config.train_target:
#         targets_list.append(mnist_dataset[digit_idx][0])

# targets_tensor = torch.empty(0)
# for idx in range(len(targets_list)):
#     digit_tensor = torchvision.transforms.ToTensor()(targets_list[idx]).squeeze()
#     flat = torch.Tensor(digit_tensor.flatten()).unsqueeze(-1)
#     targets_tensor = torch.cat((targets_tensor, flat))

# targets_tensor = targets_tensor * 2 - 1

# x_flat_multiple = torch.Tensor(mgrid.reshape(-1, 2)).repeat(len(targets_list), 1) #len(targets_list)

# #This dataset contains all target digit
# mul_dataset = torch.utils.data.TensorDataset(x_flat_multiple, targets_tensor)
# mul_train_loader = torch.utils.data.DataLoader(
#     mul_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
# )

model = SirenNet(
    dim_in=config.dim_in,
    dim_hidden=config.dim_hidden,
    dim_out=config.dim_out,
    num_layers=config.num_layers,
    w0=config.w0,
    w0_initial=config.w0_initial,
    use_bias=config.use_bias,
    final_activation=config.final_activation,
)

model_func, theta_init = functorch.make_functional(model)

########################
#STANDARD TRAINING LOOP#
########################
model_losses_adam_opt = []
opt = torch.optim.Adam(model.parameters(), lr=config.lr)
for _ in range(config.epochs):
    x, y = next(iter(train_loader))
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f'Loss: {loss.data}')
    model_losses_adam_opt.append(loss.detach().numpy())

pred = model(x_flat)
image = pred.reshape(digit_shape)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(image.detach().numpy())
axes[1].imshow(digit_tensor.detach().numpy())
fig.suptitle('Standard training')
axes[0].set_title('Prediction')
axes[1].set_title('Ground truth')
plt.savefig(filepath + 'training_result_standart.png')
plt.clf()

####################
#MEAN TRAINING LOOP#
####################
# model.set_parameters(theta_init)
# model_losses_adam_opt_mean = []
# opt = torch.optim.Adam(model.parameters(), lr=config.lr)
# for _ in range(config.epochs):
#     x, y = next(iter(mean_train_loader))
#     y_pred = model(x)
#     loss = F.mse_loss(y_pred, y)

#     opt.zero_grad()
#     loss.backward()
#     opt.step()

#     print(f'Loss: {loss.data}')
#     model_losses_adam_opt_mean.append(loss.detach().numpy())

# pred = model(x_flat)
# image = pred.reshape(digit_shape)
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(image.detach().numpy())
# axes[1].imshow(mean_digit_tensor.detach().numpy())
# fig.suptitle('Mean training')
# axes[0].set_title('Prediction')
# axes[1].set_title('Ground truth')
# plt.savefig(filepath + 'training_result_mean.png')
# plt.clf()

# model_func, theta_mean = functorch.make_functional(model)

#########################
#OPTIMIZER TRAINING LOOP#
#########################
if config.opt_type == 'LSTM':
    optimizer_list = [
        Optimizer(input_shape=parameter.shape, hidden_size=np.prod(parameter.shape) * 10)
        for parameter in theta_init
    ]
    opt_optimizer = torch.optim.Adam([p for g in optimizer_list for p in g.parameters()], lr=config.lr)

if config.opt_type == 'conv':
    optimizer_list = []
    for parameter in theta_init:
        if np.prod(parameter.shape) > (parameter.shape[0] * 3):
            optimizer_list.append(ConvOptimizer(parameter, channels=config.conv_channels, activation_func=None))
        else:
            optimizer_list.append(Optimizer(input_shape=parameter.shape, hidden_size=np.prod(parameter.shape) * 10))

    opt_optimizer = torch.optim.Adam([p for g in optimizer_list for p in g.parameters()], lr=config.lr)

log_outer_losses = []

for epoch in range(config.outer_loop_it):
    theta = list(theta_init)
    # theta = list(theta_mean)
    log_model_losses = []
    for optimizer in optimizer_list:
        optimizer.reset_state()
    for it in range(config.inner_loop_it):
        # x, y = next(iter(train_loader))
        x, y = next(iter(train_loader))
        y_pred = model_func(theta, x)
        model_loss = F.mse_loss(y, y_pred)
        print(model_loss)
        theta_gradients = torch.autograd.grad(
            model_loss, theta, create_graph=True, allow_unused=True
        )

        for i in range(len(theta)):
            if isinstance(optimizer_list[i], Optimizer):
                theta_update = optimizer_list[i](
                    theta_gradients[i]
                    .reshape(1, np.prod(theta_gradients[i].shape))
                    .detach()
                )
            else:
                theta_update = optimizer_list[i](theta_gradients[i].unsqueeze(0).detach())
            theta_update = theta_update.reshape(theta[i].shape)
            theta[i] = theta[i] - theta_update * 0.001

        if it == 0:
            outer_loss = model_loss
        else:
            outer_loss = (
                outer_loss + ((it + 1) / config.inner_loop_it) ** 2 * model_loss
            )  # Tests show that weighted the later loss stronger result in better convergence. We indeed want to aim for later loss
            # outer_loss = outer_loss + model_loss
        log_model_losses.append(model_loss.detach().numpy())
        log_outer_losses.append(outer_loss.detach().numpy())

    opt_optimizer.zero_grad()
    outer_loss.backward()
    opt_optimizer.step()
    

    print(f'outer loss: {outer_loss.data}, model_loss: {model_loss.data}')

#######################
#TRAINING LOOP AFTER G#
#######################
model.set_parameters(theta)
opt = torch.optim.Adam(model.parameters(), lr=config.lr) 

for _ in range(config.epochs):
    x, y = next(iter(train_loader))
    y_pred = model(x)
    loss = F.mse_loss(y, y_pred)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f'Loss after G: {loss.data}')
    log_model_losses.append(loss.detach().numpy())

#one prediction on training set
pred = model(x_flat)
image = pred.reshape(digit_shape)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(image.detach().numpy())
axes[1].imshow(digit_tensor.detach().numpy())
fig.suptitle('Learning to learn')
axes[0].set_title('Prediction')
axes[1].set_title('Ground truth')
plt.savefig(filepath + 'training_result_G_opt.png')
plt.clf()

fig1 = plt.plot(range(config.epochs), model_losses_adam_opt, label='Standard training')
fig2 = plt.plot(range(config.inner_loop_it + config.epochs), log_model_losses, label='G Optimized training')
legend = plt.legend(['Standard training', 'Optimized training'])
plt.title('Standard vs optimized training')
plt.savefig(filepath + 'Losses_sdt_opt.png')
plt.clf()

#################################################
#TRAINING LOOP WITH INITIALIZATION BUT WITHOUT G# 
#################################################
# model.set_parameters(theta_mean)
# opt = torch.optim.Adam(model.parameters(), lr=config.lr) 

# log_model_losses_init = []
# for _ in range(config.epochs):
#     x, y = next(iter(train_loader))
#     y_pred = model(x)
#     loss = F.mse_loss(y, y_pred)

#     opt.zero_grad()
#     loss.backward()
#     opt.step()

#     print(f'Loss after init: {loss.data}')
#     log_model_losses_init.append(loss.detach().numpy())

# #one prediction on training set
# pred = model(x_flat)
# image = pred.reshape(digit_shape)
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(image.detach().numpy())
# axes[1].imshow(digit_tensor.detach().numpy())
# fig.suptitle('Mean initialization')
# axes[0].set_title('Prediction')
# axes[1].set_title('Ground truth')
# plt.savefig(filepath + 'training_result_init_opt.png')
# plt.clf()

# fig1 = plt.plot(range(len(log_model_losses_init)), log_model_losses_init, label='Init optimized')
# fig2 = plt.plot(range(len(log_model_losses)), log_model_losses, label='G Optimized training')
# legend = plt.legend(['Init optimized', 'G Optimized training'])
# plt.title('init vs G optimized training')
# plt.savefig(filepath + 'init_vs_Ginit_losses.png')
# plt.clf()


####################
#Transfert Learning#
####################
#dataset preparation
for digit_idx, target in enumerate(mnist_dataset.targets):
    if int(target) in config.test_target:
        digit = mnist_dataset[digit_idx][0] # a PIL image  
        break 

#In case you arrive at the end of dataset
if digit is None:
    for digit_idx, target in enumerate(mnist_dataset.targets[digit_idx + 1:], start=digit_idx + 1):
        if int(target) in config.test_target:
            digit = mnist_dataset[digit_idx][0] # a PIL image  
            break 

digit_tensor = torchvision.transforms.ToTensor()(digit).squeeze()
digit_tensor = digit_tensor * 2 - 1

x_flat = torch.Tensor(mgrid.reshape(-1, 2))
y_flat = torch.Tensor(digit_tensor.flatten()).unsqueeze(-1)

dataset = torch.utils.data.TensorDataset(x_flat, y_flat)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
)

#pre-trained model version
model.set_parameters(theta)
opt = torch.optim.Adam(model.parameters(), lr=config.lr)

transfert_losses = []
for _ in range(config.epochs):
    x, y = next(iter(test_loader))
    y_pred = model(x)
    loss = F.mse_loss(y, y_pred)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f'Transfert Loss: {loss.data}')
    transfert_losses.append(loss.detach().numpy())

# #use of pre trained LSTM
# transfert_losses = []
# model.set_parameters(theta_init)
# optimizer.reset_state()
# for it in range(config.inner_loop_it):
#     x, y = next(iter(test_loader))
#     y_pred = model_func(theta, x)
#     model_loss = F.mse_loss(y, y_pred)
#     print(model_loss)
#     theta_gradients = torch.autograd.grad(
#         model_loss, theta, create_graph=True, allow_unused=True
#     )
#     for i in range(len(theta)):
#         if isinstance(optimizer_list[i], Optimizer):
#             theta_update = optimizer_list[i](
#                 theta_gradients[i]
#                 .reshape(1, np.prod(theta_gradients[i].shape))
#                 .detach()
#             )
#         else:
#             theta_update = optimizer_list[i](theta_gradients[i].unsqueeze(0).detach())
#         theta_update = theta_update.reshape(theta[i].shape)
#         theta[i] = theta[i] - theta_update

#     if it == 0:
#         outer_loss = model_loss
#     else:
#         outer_loss = (
#             outer_loss + ((it + 1) / config.inner_loop_it) ** 2 * model_loss
#         )  # Tests show that weighted the later loss stronger result in better convergence. We indeed want to aim for later loss
#         # outer_loss = outer_loss + model_loss
#     transfert_losses.append(model_loss.detach().numpy())

# model.set_parameters(theta)
# opt = torch.optim.Adam(model.parameters(), lr=config.lr * 10)

# for _ in range(config.epochs):
#     x, y = next(iter(test_loader))
#     y_pred = model(x)
#     loss = F.mse_loss(y, y_pred)

#     opt.zero_grad()
#     loss.backward()
#     opt.step()

#     print(f'Transfert Loss: {loss.data}')
#     transfert_losses.append(loss.detach().numpy())

#one prediction on training set
pred = model(x_flat)
image = pred.reshape(digit_shape)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(image.detach().numpy())
axes[1].imshow(digit_tensor.detach().numpy())
fig.suptitle('Transfert training')
axes[0].set_title('Prediction')
axes[1].set_title('Ground truth')
plt.savefig(filepath + 'training_result_transfert.png')

#Classical loop for comparison
model.set_parameters(theta_init)
opt = torch.optim.Adam(model.parameters(), lr=config.lr)

losses = []
for _ in range(config.epochs):
    x, y = next(iter(test_loader))
    y_pred = model(x)
    loss = F.mse_loss(y, y_pred)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f'Classic Loss: {loss.data}')
    losses.append(loss.detach().numpy())

#one prediction on training set
pred = model(x_flat)
image = pred.reshape(digit_shape)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(image.detach().numpy())
axes[1].imshow(digit_tensor.detach().numpy())
fig.suptitle('No transfert training')
axes[0].set_title('Prediction')
axes[1].set_title('Ground truth')
plt.savefig(filepath + 'training_result_no_transfert.png')
plt.clf()

fig1 = plt.plot(range(len(losses)), losses, label='Standard training')
fig2 = plt.plot(range(len(transfert_losses)), transfert_losses, label='Transfert training')
legend = plt.legend(['Standard training', 'Transfert training'])
plt.title('Standard vs optimized training')
plt.savefig(filepath + 'Losses_transfert.png')
plt.clf()
config.export_to_txt(filepath)

###################################
#Training initialization without G#
###################################
noG_init_losses = []
model.set_parameters(theta_init)
opt = torch.optim.Adam(model.parameters(), lr=config.lr)
for _ in range(25):
    x, y = next(iter(train_loader))
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f'Loss: {loss.data}')
    noG_init_losses.append(loss.detach().numpy())

opt = torch.optim.Adam(model.parameters(), lr=config.lr) #* 10
for _ in range(config.epochs):
    x, y = next(iter(test_loader))
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f'Loss: {loss.data}')
    noG_init_losses.append(loss.detach().numpy())

pred = model(x_flat)
image = pred.reshape(digit_shape)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(image.detach().numpy())
axes[1].imshow(digit_tensor.detach().numpy())
fig.suptitle('NoG Training')
axes[0].set_title('Prediction')
axes[1].set_title('Ground truth')
plt.savefig(filepath + 'noG_result_standart.png')
plt.clf()

fig1 = plt.plot(range(len(losses)), losses, label='Standard training')
fig2 = plt.plot(range(len(transfert_losses)), transfert_losses, label='Transfert training')
fig3 = plt.plot(range(len(noG_init_losses)), noG_init_losses, label='NoG init training')
legend = plt.legend(['Standard training', 'Transfert training', 'NoG_init'])
plt.title('Standard vs optimized training vs noG init')
plt.savefig(filepath + 'Losses_transfert.png')
plt.clf()

#memory footprint
mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
mem = mem_params + mem_bufs # in bytes

mem_params += sum([param.nelement()*param.element_size()for mod in optimizer_list for param in mod.parameters()])
mem_bufs += sum([buf.nelement()*buf.element_size()for mod in optimizer_list for buf in mod.buffers()])
mem += (mem_params + mem_bufs) # in bytes
