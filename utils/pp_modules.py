"""
A PyTorch re-implementation of the following notebook:

https://github.com/deepmind/deepmind-research/blob/master/powerpropagation/powerpropagation.ipynb


https://cs230.stanford.edu/blog/pytorch/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PowerPropLinear(nn.Module):
    """Powerpropagation Linear module."""

    def __init__(self, alpha, in_features, out_features):
        super(PowerPropLinear, self).__init__()
        self.alpha = alpha
        self.w = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.b = torch.nn.Parameter(torch.empty(out_features))

    def get_weights(self):
        return torch.sign(self.w) * torch.pow(torch.abs(self.w), self.alpha)

    def forward(self, inputs, mask=None):

        weights = self.w * torch.pow(torch.abs(self.w), self.alpha - 1)

        if mask is not None:
            weights *= mask

        outputs = F.linear(inputs, weights, self.b)

        return outputs


class MLP(nn.Module):
    """A multi-layer perceptron module."""
    def __init__(self, alpha, input_sizes=(784, 300, 100), output_sizes=(300, 100, 10)):
        super(MLP, self).__init__()
        self.alpha = alpha
        self._layers = nn.ModuleList()
        for in_features, out_features in zip(input_sizes, output_sizes):
            self._layers.append(
                PowerPropLinear(
                    alpha=alpha,
                    in_features=in_features,
                    out_features=out_features
                )
            )

    def get_weights(self):
        return [layer.get_weights().numpy() for layer in self._layers]

    def forward(self, inputs, masks=None):
        num_layers = len(self._layers)

        for i, layer in enumerate(self._layers):
            if masks is not None:
                inputs = layer(inputs, masks[i])
            else:
                inputs = layer(inputs)

            if i < (num_layers - 1):
                inputs = F.relu(inputs)

        return inputs
