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

    torch_grads = []
    for i in range(6):
        if i % 2 == 0:  # bias updates
            torch_grads.append(model._layers[i // 2].b.grad.numpy())
        else:  # weight updates
            torch_grads.append(model._layers[i // 2].w.grad.numpy().T)

    return {'acc': acc.item(), 'loss': loss.item()}, torch_grads


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
        cls.alpha = 1.0
        cls.learning_rate = 0.1

        x = np.load(r'./test_data/sample_x.npy')
        cls.x = x.reshape([100, 784]).astype(np.float32) / 255.0

        y = np.load(r'./test_data/sample_y.npy')
        cls.y = y.astype(np.int64)

        cls.x_tf = tf.convert_to_tensor(cls.x)
        cls.y_tf = tf.convert_to_tensor(cls.y)
        cls.x_torch = torch.from_numpy(cls.x)
        cls.y_torch = torch.from_numpy(cls.y)

        cls.model = TorchMLP(alpha=cls.alpha)

        weight_init = PowerPropVarianceScaling(cls.alpha)
        cls.truth = DensityNetwork(MLP(alpha=cls.alpha, w_init=weight_init))

    def setUp(self) -> None:
        # initialize ground-truth model weights
        self.truth(self.x)

        # initialize pytorch variables (set equal to TF variables)
        tf_weights = self.truth.get_weights()
        with torch.no_grad():
            for i, layer in enumerate(self.model._layers):
                layer.w = torch.nn.Parameter(torch.from_numpy(tf_weights[i].T))
                layer.b.data.zero_()

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

    def test_biases_zero(self):
        for i in range(len(self.model._layers)):
            self.assertTrue(
                (self.model._layers[i].b.detach().numpy() == 0).all(),
                msg=f'Layer {i} biases are not all zero.'
            )

    def test_grads_matching(self):
        tf_stats, tf_grads = tf_train(self.truth, self.x_tf, self.y_tf, self.tf_optim)
        torch_stats, torch_grads = torch_train(self.model, self.x_torch, self.y_torch, self.torch_optim, self.torch_loss)

        self.assertAlmostEqual(tf_stats['acc'].numpy(), torch_stats['acc'], msg='Accuracies do not match')
        self.assertAlmostEqual(tf_stats['loss'].numpy(), torch_stats['loss'], msg='Losses do not match')

        for i, (actual_grad, expected_grad) in enumerate(zip(tf_grads, torch_grads)):
            np.testing.assert_allclose(
                actual_grad.numpy(),
                expected_grad,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f'Layer {i // 2} {"bias" if i % 2 == 0 else "weight"} gradients do not match.'
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
