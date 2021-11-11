import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from modules import PowerPropLinear


def init_weights(module: nn.Module):
    if isinstance(module, PowerPropLinear):
        fan_in = calculate_fan_in(module.w.data)

        std = np.sqrt(1. / fan_in)
        a, b = -2. * std, 2. * std

        u = nn.init.trunc_normal_(module.w.data, std=std, a=a, b=b)
        u = torch.sign(u) * torch.pow(torch.abs(u), 1.0 / module.alpha)

        module.w.data = u
        module.b.data.zero_()


def calculate_fan_in(tensor: torch.Tensor):
    """Modified from: https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py"""
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if dimensions > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size

    return fan_in


def cat_loss(outputs: torch.Tensor, targets: torch.Tensor):
    """ Loss Function """
    dist = Categorical(logits=outputs)
    loss = -torch.mean(dist.log_prob(targets))
    return loss


def accuracy(outputs: torch.Tensor, targets: torch.Tensor):
    """ Accuracy """
    acc = torch.argmax(outputs, dim=1) == targets
    return torch.mean(acc.float())
