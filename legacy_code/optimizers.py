"""
Home made optimizers for metalearning appoaches. To be used with meta learning scripts
"""

import argparse
import copy
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
# import pytorch_lightning as pl
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
# import functorch
from torch.autograd import Variable
from torch.nn import functional as F

import torchvision
from einops import rearrange


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


class ConvOptimizer(nn.Module):

    """
    Marche que pour couche du centre, only 2D si tu fais une conv par couche, essayer full stack parameters ?
    """

    def __init__(self, input, channels=[32, 32, 32], activation_func=None) -> None:
        super().__init__()
        # Build the layer system
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
            in_channels=channels[-1], out_channels=1, kernel_size=3, stride=1, padding=1
        )
        layers.append(last_layer)
        self.model = nn.Sequential(*layers)

    def reset_state(self,):
        pass
        # for parameter in self.parameters():
        #     parameter.data = torch.randn(parameter.shape) * 0.01
        # #TODO: Better initialisation using noraml distribution ? See: https://pytorch.org/docs/stable/nn.init.html
        return None

    def forward(self, x):
        return self.model(x)
