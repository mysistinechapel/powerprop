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
    training_steps = 10000
    interval = 2500
    losses = []
    accs = []

    metrics = MetricsContainer(batch_size=60)
    train_iterator = iter(data_loader)
    for idx in range(training_steps + 1):
        try:
            (data, target) = next(train_iterator)
        except StopIteration:
            train_iterator = iter(data_loader)
            (data, target) = next(train_iterator)

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
            print(f'Interval: [{idx // interval}][{idx}/{training_steps}]\t'
                  f'Loss={loss.item():.4f} ({metrics.avg_loss:.4f})\t'
                  f'Acc={acc.item():.4f} ({metrics.avg_acc:.4f})')


def evaluate(model, inputs, targets, criterion, metric=accuracy):
    with torch.no_grad():
        out = model(inputs)
        loss = criterion(out, targets)
        acc = metric(out, targets)

    print(f'Test Set:\tLoss={loss.item():.4f}\tAcc={acc.item():.4f}')


def evaluate_pruning(model, test_x, test_y, criterion, metric=accuracy):
    model.requires_grad = False
    final_weights = model.get_weights()
    print("FINAL WEIGHTS:", final_weights)
    eval_at_sparsity_level = np.geomspace(0.01, 1.0, 20).tolist()

    # Deepmind notebook took in a list of models for each alpha.
    # This initial iteration just looks at the one model
    models = [model]
    n_models = len(models)
    acc_at_sparsity = [[] for _ in range(n_models)]
    alphas = [1.0]
    # Half the sparsity at output layer
    for p_to_use in eval_at_sparsity_level:
        percent = 2 * [p_to_use] + [min(1.0, p_to_use * 2)]
        masks = []
        for i, w in enumerate(final_weights):
            masks.append(prune_by_magnitude(percent[i], w))

        out = model(test_x, masks=masks)
        loss = criterion(out, test_y)
        acc = accuracy(out, test_y)

        acc_at_sparsity.append(acc.item())
        print(' Performance @ {:1.0f}% of weights [Alpha {}]: Acc {:1.3f} Loss {:1.3f} '.format(
            100 * p_to_use, alphas, acc.item(), loss.item()))


def prune_by_magnitude(percent_to_keep, weight):
    mask = _bottom_k_mask(percent_to_keep, np.abs(weight.flatten()))
    return mask.reshape(weight.shape)


def _bottom_k_mask(percent_to_keep, condition):
    how_many = int(percent_to_keep * condition.size)
    top_k = torch.topk(torch.as_tensor(condition), k=how_many)
    mask = torch.zeros(condition.shape)
    mask[top_k.indices] = 1

    assert torch.sum(mask) == how_many

    return mask
