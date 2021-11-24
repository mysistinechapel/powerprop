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
        return [layer.get_weights().detach().numpy() for layer in self._layers]

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

    def loss(self, inputs, targets, masks=None):
        with torch.no_grad():
            outputs = self.forward(inputs, masks)
            dist = torch.distributions.categorical.Categorical(logits=outputs)
            loss = -torch.mean(dist.log_prob(targets))

        accuracy = torch.sum(targets == torch.argmax(outputs, axis=1)) / targets.shape[0]

        return loss, {'loss': loss, 'acc': accuracy}


class PowerPropConv(nn.Module):
    """Powerpropagation Conv2D module."""

    def __init__(self, alpha, in_channels, out_channels, kernel_size=3):
        super(PowerPropConv, self).__init__()
        self.alpha = alpha
        self.w = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.b = torch.nn.Parameter(torch.empty(out_channels))

    def get_weights(self):
        return torch.sign(self.w) * torch.pow(torch.abs(self.w), self.alpha)

    def forward(self, inputs, mask=None):

        weights = self.w * torch.pow(torch.abs(self.w), self.alpha - 1)

        if mask is not None:
            weights *= mask

        outputs = F.conv2d(inputs, weights, self.b)

        return outputs


class CNN(nn.Module):
    """A convolutional neural network module."""
    def __init__(self, alpha, in_channels=3, out_channels=(64, 128, 256)):
        super(CNN, self).__init__()
        self.alpha = alpha
        self.conv_layers = nn.ModuleDict()
        self.fc_layers = nn.ModuleList()

        # create convolutional layers
        for i, channels in enumerate(out_channels):
            self.conv_layers[f'ConvLayer{2*i + 1}'] = PowerPropConv(
                alpha=alpha, in_channels=in_channels, out_channels=channels
            )
            self.conv_layers[f'ConvLayer{2*i + 2}'] = PowerPropConv(
                alpha=alpha, in_channels=channels, out_channels=channels
            )
            in_channels = channels

        # create FC layers
        in_features = 512  # TODO update with expected number
        for i, features in enumerate((256, 256, 10)):
            self.fc_layers.append(
                PowerPropLinear(alpha=alpha, in_features=in_features, out_features=features)
            )
            in_features = features

    def get_weights(self):
        weights = [layer.get_weights().numpy() for layer in self.conv_layers.values()]
        weights.extend([layer.get_weights().numpy() for layer in self.fc_layers])
        return weights

    def forward(self, inputs, masks=None):
        for i in range(len(self.conv_layers)):
            layer = self._layers[f'ConvLayer{i + 1}']
            if masks is not None:
                inputs = layer(inputs, masks[i])
            else:
                inputs = layer(inputs)

            # if first in conv pair, apply relu, otherwise apply max pool at end of pair
            if i % 2 == 0:
                inputs = F.relu(inputs)
            else:
                inputs = F.max_pool2d(inputs, kernel_size=2, stride=2)

        num_fcs = len(self.fc_layers)
        for i, layer in enumerate(self.fc_layers):
            if masks is not None:
                inputs = layer(inputs, masks[i])
            else:
                inputs = layer(inputs)

            if i < (num_fcs - 1):
                inputs = F.relu(inputs)

        return inputs
