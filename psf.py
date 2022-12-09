import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.font_manager import X11FontDirectories

from models import SirenNet

x = np.linspace(-10, 10, 1000)


def foo(x):
    return np.sin(2 * x)


fig = plt.plot(x, foo(x))
plt.show()


def psf(x_0, x):

    FHWM = (
        1.0
    )  # FHWM is equal to the slice thikness (for ssFSE sequence), cf article Jiang et al.
    sigma = FHWM / (2 * np.sqrt((2 * np.log(2))))
    res = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        (-(x - x_0) ** 2) / (2 * (sigma ** 2))
    )

    return res


def gauss(x, height=0.8, center=0, width=0.2):
    return height * np.exp(-(x - center) ** 2 / (2 * width * width))


x = np.linspace(-10, 10, 1000)
fig = plt.plot(x, gauss(x))
plt.show()

y = np.linspace(-10, 10, 10)


def discretization(x):
    return foo(np.sum(gauss(np.linspace(-1, 1, 11), center=x), axis=0))


fig = plt.plot(y, discretization(y))
plt.show()
