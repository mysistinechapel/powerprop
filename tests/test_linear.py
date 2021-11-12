import numpy as np

import torch
import unittest

from utils.pp_modules import PowerPropLinear
from utils.utils import init_weights


class TestLinearLayers(unittest.TestCase):

    def setUp(self) -> None:
        self.alpha = 1.0
        self.in_feats = 784
        self.out_feats = 300
        self.fc = PowerPropLinear(alpha=self.alpha, in_features=self.in_feats, out_features=self.out_feats)
        self.fc.apply(init_weights)

    def test_biases(self):
        expected = np.zeros(self.out_feats)
        np.testing.assert_array_equal(expected, self.fc.b.data.numpy())
        self.assertEqual(self.out_feats, self.fc.b.shape)

    def test_weights(self):
        expected_mean = 0.0
        actual_mean = torch.mean(self.fc.w.data).item()
        diff = (expected_mean - actual_mean) ** 2
        self.assertAlmostEqual(expected_mean, diff)
        self.assertTupleEqual((self.out_feats, self.in_feats), self.fc.w.shape)


if __name__ == '__main__':
    unittest.main(verbosity=2)
