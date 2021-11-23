import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from utils.metrics import accuracy, MetricsContainer
from utils.pp_modules import PowerPropLinear, PowerPropConv


def preprocess(data: torch.Tensor):
    """
    Preprocess MNIST Data
    """
    data = data.flatten(start_dim=1)
    data = data.float() / 255.
    return data


def train_val_split(data, val_size: int):
    """
    Split dataset into training and validation sets.

    Validation set is taken as last 'val_size' records in data.
    """
    train_x = data.data[:-val_size]
    train_y = data.targets[:-val_size]

    valid_x = data.data[-val_size:]
    valid_y = data.targets[-val_size:]

    return train_x, train_y, valid_x, valid_y


def init_weights(module: nn.Module):
    if isinstance(module, (PowerPropLinear, PowerPropConv)):
        fan_in = calculate_fan_in(module.w.data)

        # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        distribution_stddev = .87962566103423978

        std = np.sqrt(1. / fan_in) / distribution_stddev
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
    """ Loss Function (another way to compute cross-entropy) """
    dist = Categorical(logits=outputs)
    loss = -torch.mean(dist.log_prob(targets))
    return loss


def train(model, data_loader, optimizer, criterion, metric=accuracy):

    # TODO dynamic epoch and interval
    epoch = 0
    interval = 100
    losses = []
    accs = []

    metrics = MetricsContainer(batch_size=60)

    for idx, (data, target) in enumerate(data_loader):

        optimizer.zero_grad()

        out = model(data)
        loss = criterion(out, target)
        loss.backward()

        optimizer.step()

        acc = metric(out, target)
        losses.append(loss.item())
        accs.append(acc.item())

        metrics.update(loss.item(), acc.item())

        if idx % interval == 0:
            print(f'Epoch: [{epoch}][{idx}/{len(data_loader)}]\t'
                  f'Loss={loss.item():.4f} ({metrics.avg_loss:.4f})\t'
                  f'Acc={acc.item():.4f} ({metrics.avg_acc:.4f})')


def evaluate(model, inputs, targets, criterion, metric=accuracy):

    with torch.no_grad():
        out = model(inputs)
        loss = criterion(out, targets)
        acc = metric(out, targets)

    print(f'Test Set:\tLoss={loss.item():.4f}\tAcc={acc.item():.4f}')



#
# print(('Epoch: [{0}][{1}/{2}]\t'
#                    'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
#                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                    'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
#                   .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses, top1=acc))