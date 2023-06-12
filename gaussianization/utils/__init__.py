
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import warnings
from typing import Callable
from contextlib import contextmanager


from .max_kswd import maxKSWDdirection as max_kswd


def _except(f: Callable, x: torch.Tensor, *dim, **kwargs):
    """ Apply f on all dimensions except those specified in dim """
    result = x
    dimensions = [d for d in range(x.dim()) if d not in dim]

    if not dimensions:
        raise ValueError(f"Cannot exclude dims {dim} from x with shape {x.shape}: No dimensions left.")

    return f(result, dim=dimensions, **kwargs)


def sum_except(x: torch.Tensor, *dim):
    """ Sum all dimensions of x except the ones specified in dim """
    return _except(torch.sum, x, *dim)


def sum_except_batch(x):
    """ Sum all dimensions of x except the batch dimension """
    return sum_except(x, 0)


def mean_except(x: torch.Tensor, *dim):
    """ See sum_except """
    return _except(torch.mean, x, *dim)


def mean_except_batch(x):
    """ See sum_except_batch """
    return mean_except(x, 0)


def std_except(x: torch.Tensor, *dim):
    return _except(torch.std, x, *dim)


def std_except_batch(x):
    return std_except(x, 0)


def norm_except(x, *dim):
    return _except(torch.norm, x, *dim)


def norm_except_batch(x):
    return norm_except(x, 0)


def repeat_as(x1: torch.Tensor, x2: torch.Tensor):
    """ Repeat x1 to match the shape of x2 """
    if x1.dim() != x2.dim():
        raise RuntimeError(f"Tensors must have matching dimension.")

    s1 = torch.tensor(x1.shape)
    s2 = torch.tensor(x2.shape)

    div = s2 // s1
    mod = s2 % s1
    if torch.any(torch.nonzero(mod)):
        raise RuntimeError(f"Cannot repeat tensor of shape {x1.shape} to match {x2.shape}.")

    return x1.repeat(div.tolist())


def repeat_dim(x: torch.Tensor, count: int, *, dim: int):
    s = torch.ones(x.dim(), dtype=torch.int32)
    s[dim] = count

    return x.repeat(*s.tolist())


def unsqueeze_to(x: torch.Tensor, dim: int, side="right"):
    """ Unsqueeze x1 on the right to match the given dimensionality """
    if dim < x.dim():
        raise RuntimeError(f"Cannot unsqueeze tensor of dim {x.dim()} to {dim}.")

    idx = [None] * (dim - x.dim())
    if side == "right":
        idx = [..., *idx]
    elif side == "left":
        idx = [*idx, ...]
    else:
        raise ValueError(f"Unknown side: {side}")

    return x[idx]


def unsqueeze_as(x1: torch.Tensor, x2: torch.Tensor, **kwargs):
    return unsqueeze_to(x1, x2.dim(), **kwargs)


def normalize(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    mean = torch.mean(x, dim=dim, keepdim=True)
    std = torch.std(x, dim=dim, keepdim=True)

    return (x - mean) / std


def augment(points: torch.Tensor, noise: float = 0.05) -> torch.Tensor:
    noise = noise * torch.randn_like(points)

    return noise + points


def get_activation(activation):
    """ Return the corresponding torch Activation function by string """
    match activation.lower():
        case "relu":
            return nn.ReLU
        case "elu":
            return nn.ELU
        case "selu":
            return nn.SELU
        case activation:
            raise NotImplementedError(f"Unsupported Activation: {activation}")


def make_dense(widths: list[int], activation: str, dropout: float = None):
    """ Make a Dense Network from given layer widths and activation function """
    if len(widths) < 2:
        raise ValueError(f"Need at least Input and Output Layer.")
    elif len(widths) < 3:
        warnings.warn(f"Should use more than zero hidden layers.")

    Activation = get_activation(activation)

    network = nn.Sequential()

    # input is x, time, condition
    input_layer = nn.Linear(in_features=widths[0], out_features=widths[1])
    network.add_module("Input_Layer", input_layer)
    network.add_module("Input_Activation", Activation())

    for i in range(1, len(widths) - 2):
        if dropout is not None:
            network.add_module(f"Dropout_{i}", nn.Dropout1d(p=dropout))
        hidden_layer = nn.Linear(in_features=widths[i], out_features=widths[i + 1])
        network.add_module(f"Hidden_Layer_{i}", hidden_layer)
        network.add_module(f"Hidden_Activation_{i}", Activation())

    # output is velocity
    output_layer = nn.Linear(in_features=widths[-2], out_features=widths[-1])
    network.add_module("Output_Layer", output_layer)

    return network


def inverse_softplus(x: torch.Tensor):
    return torch.log(torch.exp(x) - 1)


def shifted_softplus(x: torch.Tensor):
    """ shifted softplus such that f(0) = 1 """
    shift = np.log(np.e - 1)
    return F.softplus(x + shift)


def inverse_shifted_softplus(x: torch.Tensor):
    shift = np.log(np.e - 1)
    return inverse_softplus(x) - shift


@contextmanager
def separate(title: str = "", marker: str = "%", length: int = 64):
    top = marker * (length // len(marker))
    bottom = marker * (length // len(marker))

    start = length // 2 - len(title) // 2 - 1
    end = length // 2 + len(title) // 2 + 1

    top = " ".join([top[:start], title, top[end:]])
    top = top[:length]

    print(top)
    yield
    print(bottom)

