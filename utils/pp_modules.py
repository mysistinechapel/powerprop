"""
A PyTorch re-implementation of the following notebook:

https://github.com/deepmind/deepmind-research/blob/master/powerpropagation/powerpropagation.ipynb
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
        weights = self.w.detach()
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

    def forward(self, inputs, mask=None):
        weights = self.w * torch.pow(torch.abs(self.w), self.alpha - 1.)

        if mask is not None:
            weights *= mask

        return F.linear(inputs, weights, self.b)


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
        return [layer.get_weights() for layer in self._layers]

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

        accuracy = torch.sum(targets == torch.argmax(outputs, dim=1)) / targets.shape[0]

        return {'loss': loss, 'acc': accuracy}


class PowerPropConv(nn.Module):
    """Powerpropagation Conv2D module."""

    def __init__(self, alpha, in_channels, out_channels, kernel_size=3):
        super(PowerPropConv, self).__init__()
        self.alpha = alpha
        self.w = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.b = torch.nn.Parameter(torch.empty(out_channels))

    def get_weights(self):
        weights = self.w.detach()
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

    def forward(self, inputs, mask=None):

        weights = self.w * torch.pow(torch.abs(self.w), self.alpha - 1.)

        if mask is not None:
            weights *= mask

        return F.conv2d(inputs, weights, self.b, stride=1, padding=1)


class CNN(nn.Module):
    """A convolutional neural network module."""
    def __init__(self, alpha, in_channels=3, out_channels=(64, 128, 256)):
        super(CNN, self).__init__()
        self.alpha = alpha
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # create convolutional layers
        for channels in out_channels:
            self.conv_layers.append(
                PowerPropConv(alpha=alpha, in_channels=in_channels, out_channels=channels)
            )
            self.conv_layers.append(
                PowerPropConv(alpha=alpha, in_channels=channels, out_channels=channels)
            )
            self.batch_norms.extend([
                nn.BatchNorm2d(channels), nn.BatchNorm2d(channels)
            ])
            in_channels = channels

        # create FC layers
        in_features = 4096
        for features in (256, 256, 10):
            self.fc_layers.append(
                PowerPropLinear(alpha=alpha, in_features=in_features, out_features=features)
            )
            in_features = features

        self.num_fcs = len(self.fc_layers)
        self.num_convs = len(self.conv_layers)

    def get_weights(self):
        weights = [layer.get_weights() for layer in self.conv_layers]
        weights.extend([layer.get_weights() for layer in self.fc_layers])
        return weights

    def forward(self, inputs, masks=None):
        if masks is None:
            masks = [None] * (self.num_convs + self.num_fcs)

        for i, layer in enumerate(self.conv_layers):
            inputs = layer(inputs, masks[i])
            inputs = F.relu(self.batch_norms[i](inputs))
            if i % 2 == 1:  # pooling after each pair of conv layers
                inputs = F.max_pool2d(inputs, kernel_size=2, stride=2)

        inputs = torch.flatten(inputs, start_dim=1)
        for i, layer in enumerate(self.fc_layers):
            inputs = layer(inputs, masks[i + self.num_convs])
            if i < (self.num_fcs - 1):
                inputs = F.relu(inputs)

        return inputs


class PlainCNN(nn.Module):
    """A convolutional neural network module."""
    def __init__(self, in_channels=3, out_channels=(64, 128, 256)):
        super(PlainCNN, self).__init__()
        self.network = nn.Sequential()
        # create convolutional layers
        for i, channels in enumerate(out_channels):
            self.make_block(in_channels, channels, 2 * i + 1)
            self.make_block(channels, channels, 2 * i + 2)
            self.network.add_module(
                f'MaxPool{i + 1}',
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            in_channels = channels

        self.network.add_module('Flatten', nn.Flatten(start_dim=1))
        # create FC layers
        in_features = 4096
        for i, features in enumerate([256, 256, 10]):
            self.network.add_module(
                f'Linear{i + 1}',
                nn.Linear(in_features=in_features, out_features=features)
            )
            if i < 2:
                self.network.add_module(
                    f'ReLU{i + 7}',
                    nn.ReLU()
                )
            in_features = features

    def make_block(self, in_channels, channels, idx):
        self.network.add_module(
            f'Conv{idx}',
            nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        )
        self.network.add_module(f'BatchNorm{idx}', nn.BatchNorm2d(channels))
        self.network.add_module(f'ReLU{idx}', nn.ReLU())

    def forward(self, inputs):
        return self.network(inputs)
