import numpy as np
import sonnet as snt
import tensorflow as tf

import torch
import unittest

from utils.pp_modules import MLP as TorchMLP
from utils.tf_modules import DensityNetwork, MLP, PowerPropVarianceScaling


def torch_train(model, train_x, train_y, optimizer, loss_fn):
    optimizer.zero_grad()

    out = model(train_x)
    loss = loss_fn(out, train_y)
    loss.backward()

    optimizer.step()

    acc = (torch.argmax(out, dim=1) == train_y).float()
    acc = torch.mean(acc)
    return {'acc': acc.item(), 'loss': loss.item()}


@tf.function
def tf_train(model, train_x, train_y, optimizer):
    with tf.GradientTape() as tape:
        loss, stats = model.loss(train_x, train_y)

    train_vars = model.trainable_variables()
    train_grads = tape.gradient(loss, train_vars)
    optimizer.apply(train_grads, train_vars)

    return stats, train_grads


class TestBackProp(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.learning_rate = 0.1

        x = np.load(r'./test_data/sample_x.npy')
        cls.x = x.reshape([100, 784]).astype(np.float32) / 255.0

        y = np.load(r'./test_data/sample_y.npy')
        cls.y = y.astype(np.int64)

        cls.x_tf = tf.convert_to_tensor(cls.x)
        cls.y_tf = tf.convert_to_tensor(cls.y)
        cls.x_torch = torch.from_numpy(cls.x)
        cls.y_torch = torch.from_numpy(cls.y)

    def setUp(self) -> None:
        self.alpha = 1.0
        self.model = TorchMLP(alpha=self.alpha)

        weight_init = PowerPropVarianceScaling(self.alpha)
        self.truth = DensityNetwork(MLP(alpha=self.alpha, w_init=weight_init))

        # initialize ground-truth model weights
        self.truth(self.x)

        # initialize pytorch variables (set equal to TF variables)
        tf_weights = self.truth.get_weights()
        with torch.no_grad():
            for i, layer in enumerate(self.model._layers):
                layer.w = torch.nn.Parameter(torch.from_numpy(tf_weights[i].T))

        self.tf_optim = snt.optimizers.SGD(learning_rate=self.learning_rate)
        self.torch_optim = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.torch_loss = torch.nn.CrossEntropyLoss()

    def test_weights_matching(self):
        n_layers = len(self.model._layers)
        tf_weights = self.truth.get_weights()

        for i in range(n_layers):
            np.testing.assert_array_equal(
                tf_weights[i].T, self.model._layers[i].w.detach().numpy(),
                err_msg=f'Layer {i} weights do not match.'
            )

    def test_grads_matching(self):

        tf_stats, tf_grads = tf_train(self.truth, self.x_tf, self.y_tf, self.tf_optim)
        torch_stats = torch_train(self.model, self.x_torch, self.y_torch, self.torch_optim, self.torch_loss)


if __name__ == '__main__':
    unittest.main(verbosity=2)
