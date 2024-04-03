import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from nn import *

@pytest.mark.parametrize(
    "inputs",
    [
        np.array([
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]
        ]).T
    ]
)

@pytest.mark.parametrize(
    "weights",
    [
        np.array([
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]
        ])
    ]
)

@pytest.mark.parametrize(
    "bias",
    [
        np.array([[2.0, 3.0, 0.5]]).T
    ]
)

class TestLinearLayer:

    def test_foward(self, inputs, weights, bias) -> None:
        expected = np.array([
            [4.8, 8.9, 1.4100000000000004],
            [1.21, -1.8100000000000005, 1.0509999999999997],
            [2.385, 0.19999999999999996, 0.025999999999999912]
        ])

        layer_1 = LinearLayer(4, 3)
        layer_1.weights = weights
        layer_1.bias = bias

        out1 = layer_1.forward(inputs)

        result_only_zeros = not np.any(expected - out1)

        assert(result_only_zeros)