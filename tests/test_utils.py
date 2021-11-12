import numpy as np

import torch
import unittest

from utils.utils import calculate_fan_in, accuracy


class TestUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.in_feats = 784
        self.fc = torch.empty(300, self.in_feats)
        self.conv = torch.empty(1, 64, 3, 3)

    def test_linear_fan(self):
        actual = calculate_fan_in(self.fc)
        self.assertEqual(self.in_feats, actual)

    def test_conv_fan(self):
        expected = 64 * 3 * 3
        actual = calculate_fan_in(self.conv)
        self.assertEqual(expected, actual)

    def test_accuracy(self):
        outputs = torch.eye(10)
        labels = torch.Tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

        acc1 = accuracy(outputs, labels)
        self.assertEqual(0.0, acc1.item())

        acc2 = accuracy(torch.fliplr(outputs), labels)
        self.assertEqual(1.0, acc2.item())


if __name__ == '__main__':
    unittest.main(verbosity=2)
