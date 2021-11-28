import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from utils.metrics import accuracy, MetricsContainer
from utils.pp_modules import PowerPropLinear, PowerPropConv


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


def train(model, data_loader, optimizer, criterion, metric=accuracy, epochs=50):
    interval = 2500
    losses = []
    accs = []
    metrics = MetricsContainer(batch_size=data_loader.batch_size)

    batches = len(data_loader)
    training_steps = epochs * batches
    for epoch in range(epochs):
        for i, (data, targets) in enumerate(data_loader, start=epoch * batches):
            optimizer.zero_grad()

            out = model(data)
            loss = criterion(out, targets)
            loss.backward()

            optimizer.step()

            acc = metric(out, targets)
            losses.append(loss.item())
            accs.append(acc.item())

            metrics.update(loss.item(), acc.item())

            if i % interval == 0:
                print(f'Epoch: [{epoch + 1}/{epochs}][{i}/{training_steps}]\t'
                      f'Loss={loss.item():.4f} ({metrics.avg_loss:.4f})\t'
                      f'Acc={acc.item():.4f} ({metrics.avg_acc:.4f})')


def evaluate(model, inputs, targets, criterion, masks=None, metric=accuracy):
    model.eval()
    with torch.no_grad():
        out = model(inputs, masks)
        loss = criterion(out, targets)
        acc = metric(out, targets)

    return loss.item(), acc.item()


def evaluate_pruning(models, test_x, test_y, alphas, criterion):
    orig_model_weights = [m.get_weights() for m in models]
    sparsity_levels = np.geomspace(0.01, 1.0, 20).tolist()
    acc_at_sparsity = [[] for _ in range(len(models))]
    all_masks = []
    for p_to_use in sparsity_levels:
        # Half the sparsity at output layer
        percent = [p_to_use, p_to_use, min(1.0, p_to_use * 2.0)]
        for m_id, model_to_use in enumerate(models):
            masks = []
            for i, w in enumerate(orig_model_weights[m_id]):
                masks.append(prune_by_magnitude(percent[i], w))

            loss, acc = evaluate(model_to_use, test_x, test_y, criterion, masks=masks)
            acc_at_sparsity[m_id].append(acc)
            all_masks.append(masks)
            print(f'Performance @ {100 * p_to_use:1.0f}% of weights [alpha={alphas[m_id]}]:\t'
                  f'Acc={acc:1.3f}\tLoss={loss:1.3f}')
    return acc_at_sparsity, sparsity_levels, all_masks, orig_model_weights


def prune_by_magnitude(percent_to_keep, weight):
    mask = _bottom_k_mask(percent_to_keep, torch.abs(weight.flatten()))
    return mask.reshape(weight.shape)


def _bottom_k_mask(percent_to_keep, condition):
    how_many = int(percent_to_keep * torch.numel(condition))
    top_k = torch.topk(torch.as_tensor(condition), k=how_many)
    mask = torch.zeros(condition.shape)
    mask[top_k.indices] = 1

    return mask


def plot_sparsity_performance(acc_at_sparsity, eval_at_sparsity_level, model_types, dataset_desc="MNIST"):
    sns.set_style("whitegrid")
    sns.set_context("paper")

    f, ax = plt.subplots(1, 1, figsize=(7, 5))
    for acc, label in zip(acc_at_sparsity, model_types):
        ax.plot(eval_at_sparsity_level, acc, label=label, marker='o', lw=2)

    ax.set_xscale('log')
    ax.set_xlim([1.0, 0.01])
    ax.set_ylim([0.0, 1.0])
    ax.legend(frameon=False)
    ax.set_xlabel('Weights Remaining (%)')
    ax.set_ylabel('Test Accuracy (%)')

    sns.despine()

    plt.savefig("images/" + dataset_desc + "_sparsity_performance.png")

def plot_pruned_vs_remaining_weights(init_weights, final_weights, chart_name="Baseline", dataset_desc="CIFAR"):
    sns.set_style("whitegrid")
    sns.set_context("paper")
    f, ax = plt.subplots(1, 1, figsize=(7, 5))
    init_weights = init_weights[0].flatten()
    final_weights = final_weights.flatten().detach().numpy()
    y = np.arange(-10, 10, .002)

    ax.set_xscale('log')
    ax.set_xlim([1.0, 0.01])

    ax.set_ylim([-10, 10])
    ax.legend( ["Pruned Weights", "Remaining Weights"])
    ax.set_xlabel('Initial Weight')
    ax.set_ylabel('Remaining Weights')


    final_weight_samples = np.random.choice(final_weights, size=10000, replace=False, p=None)
    init_weight_samples = np.random.choice(init_weights, size=10000, replace=False, p=None)

    plt.scatter(init_weight_samples, y=y, c="red")
    plt.scatter(final_weight_samples, y=y, c="blue")

    plt.savefig("images/" + chart_name + "_" + dataset_desc + ".png")


