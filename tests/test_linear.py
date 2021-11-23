import numpy as np
import tensorflow as tf

import torch
import unittest

from utils.pp_modules import PowerPropLinear as TorchLinear
from utils.tf_modules import PowerPropVarianceScaling, PowerPropLinear as TFLinear
from utils.utils import init_weights


class TestLinearLayers(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.alpha = 2.0
        cls.learning_rate = 0.1
        cls.in_feats = 784
        cls.out_feats = 300

        x = np.load(r'./test_data/sample_x.npy')
        cls.x = x.reshape([100, 784]).astype(np.float32) / 255.0

        weight_init = PowerPropVarianceScaling(cls.alpha)
        cls.tf_fc = TFLinear(output_size=cls.out_feats, alpha=cls.alpha, w_init=weight_init, name='TFLinear')
        cls.torch_fc = TorchLinear(alpha=cls.alpha, in_features=cls.in_feats, out_features=cls.out_feats)

    def setUp(self) -> None:
        self.tf_fc(tf.convert_to_tensor(self.x))
        self.torch_fc.apply(init_weights)

    def test_biases(self):
        expected = np.zeros(self.out_feats)
        np.testing.assert_array_equal(expected, self.torch_fc.b.data.numpy())
        self.assertEqual(self.out_feats, self.torch_fc.b.shape[0])

    def test_weights(self):
        expected_mean = 0.0
        actual_mean = self.torch_fc.w.mean().item()
        diff = (expected_mean - actual_mean) ** 2
        self.assertAlmostEqual(expected_mean, diff, places=6)
        self.assertTupleEqual((self.out_feats, self.in_feats), self.torch_fc.w.shape)

    def test_weight_dist(self):
        expected_val = 0.0
        expected_std = self.tf_fc.w.numpy().std()

        self.assertAlmostEqual(
            expected_val, (expected_val - self.torch_fc.w.mean().item()) ** 2, places=6,
            msg='Mean should be close to 0'
        )
        self.assertAlmostEqual(
            expected_val, (expected_std - self.torch_fc.w.std().item()) ** 2, places=6,
            msg=f'StdDev should be close to {expected_std}'
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
