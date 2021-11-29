"""
The following code is from:

Jonathan Schwarz, Siddhant M. Jayakumar, Razvan Pascanu, Peter E. Latham, and Yee Whye Teh.
Powerpropagation: A sparsity inducing weight reparameterisation.

https://github.com/deepmind/deepmind-research/blob/master/powerpropagation/powerpropagation.ipynb

We use it here only for verifying our PyTorch implementation.
"""

import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp


class PowerPropVarianceScaling(snt.initializers.VarianceScaling):

    def __init__(self, alpha, *args, **kwargs):
        super(PowerPropVarianceScaling, self).__init__(*args, **kwargs)
        self._alpha = alpha

    def __call__(self, shape, dtype):
        u = super(PowerPropVarianceScaling, self).__call__(shape, dtype).numpy()

        return tf.sign(u) * tf.pow(tf.abs(u), 1.0 / self._alpha)


class PowerPropLinear(snt.Linear):
    """Powerpropagation Linear module."""

    def __init__(self, alpha, *args, **kwargs):
        super(PowerPropLinear, self).__init__(*args, **kwargs)
        self._alpha = alpha

    def get_weights(self):
        return tf.sign(self.w) * tf.pow(tf.abs(self.w), self._alpha)

    def __call__(self, inputs, mask=None):
        self._initialize(inputs)
        params = self.w * tf.pow(tf.abs(self.w), self._alpha - 1)

        if mask is not None:
            params *= mask

        outputs = tf.matmul(inputs, params) + self.b

        return outputs


class MLP(snt.Module):
    """A multi-layer perceptron module."""

    def __init__(self, alpha, w_init, output_sizes=(300, 100, 10), name='MLP'):

        super(MLP, self).__init__(name=name)
        self._alpha = alpha
        self._w_init = w_init

        self._layers = []
        for index, output_size in enumerate(output_sizes):
            self._layers.append(
                PowerPropLinear(
                    output_size=output_size,
                    alpha=alpha,
                    w_init=w_init,
                    name="linear_{}".format(index)))

    def get_weights(self):
        return [l.get_weights().numpy() for l in self._layers]

    def __call__(self, inputs, masks=None):
        num_layers = len(self._layers)

        for i, layer in enumerate(self._layers):
            if masks is not None:
                inputs = layer(inputs, masks[i])
            else:
                inputs = layer(inputs)
            if i < (num_layers - 1):
                inputs = tf.nn.relu(inputs)

        return inputs


class DensityNetwork(snt.Module):
    """Produces categorical distribution."""

    def __init__(self, network=None, name="DensityNetwork", *args, **kwargs):
        super(DensityNetwork, self).__init__(name=name)
        self._network = network

    def __call__(self, inputs, masks=None, *args, **kwargs):
        outputs = self._network(inputs, masks, *args, **kwargs)

        return tfp.distributions.Categorical(logits=outputs), outputs

    def trainable_variables(self):
        return self._network.trainable_variables

    def get_weights(self):
        return self._network.get_weights()

    def loss(self, inputs, targets, masks=None, *args, **kwargs):
        dist, logits = self.__call__(
            inputs, masks, *args, **kwargs)
        loss = -tf.reduce_mean(dist.log_prob(targets))

        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits, axis=1), targets), tf.float32))

        return loss, {'loss': loss, 'acc': accuracy}
