import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from nn import *

np.set_printoptions(precision=40)

# Define a single fixture that returns a dictionary containing inputs, weights, and bias
@pytest.fixture
def layer_data():
    return {
        'inputs': np.array([
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]
        ]).T,
        'weights': np.array([
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]
        ]),
        'bias': np.array([[2.0, 3.0, 0.5]]).T
    }

class TestLinearLayer:
    def test_forward(self, layer_data) -> None:
        expected = np.array([
            [4.8, 8.9, 1.4100000000000004],
            [1.21, -1.8100000000000005, 1.0509999999999997],
            [2.385, 0.19999999999999996, 0.025999999999999912]
        ])

        layer_1 = LinearLayer(4, 3)
        layer_1.weights = layer_data['weights']
        layer_1.bias = layer_data['bias']

        out1 = layer_1.forward(layer_data['inputs'])

        result_only_zeros = not np.any(expected - out1)

        assert(result_only_zeros)

    def test_backward(self, layer_data) -> None:
        expected_weights = np.array([
            [0.179515, 0.742093, -0.510153, 0.971328],
            [0.5003665, -0.9152577000000001, 0.2529017, -0.5021842],
            [-0.26274600000000004, -0.27584020000000004, 0.16295920000000003, 0.8636583]
        ])

        expected_bias = np.array([
            [1.98489, 2.997739, 0.497389]
        ])

        layer_1 = LinearLayer(4, 3)
        act_1 = ActivationRelu()

        layer_1.weights = layer_data['weights']
        layer_1.bias = layer_data['bias']

        out1 = layer_1.forward(layer_data['inputs'])

        relu = act_1.forward(out1)
        drelu = act_1.backward(out1)

        dinputs = layer_1.backward(drelu)

        layer_1.weights += -0.001 * layer_1.dweights
        layer_1.bias += -0.001 * layer_1.dbias

        weights_only_zeros = not np.any(expected_weights - layer_1.weights)

        assert(weights_only_zeros)

class TestActivationSoftmaxLossCrossEntropy:
    def test_forward(self) -> None:
        # The input vector
        inputs_loggits = np.array([-2, -1, 0]).reshape(-1, 1)  # Reshaping for compatibility with the softmax function
        inputs_ytrue = np.array([1])
        
        # Expected output calculated manually or with another tool for verification
        # Softmax formula: exp(x_i) / sum(exp(x)) for each element x_i in the input vector
        expected_output = np.array([0.09003057, 0.24472847, 0.66524096]).reshape(-1, 1)  # Adjusted to match the input shape
        expected_loss = 4.222817893333141
        
        # Creating an instance of ActivationSoftmax
        softmax = ActivationSoftmaxLossCrossEntropy()
        
        # Calling the forward function
        output_loss = softmax.forward(inputs_loggits, inputs_ytrue)
        output_probabilities = softmax.output
        
        # Asserting that the outputs are close
        assert np.allclose(output_probabilities, expected_output), "The softmax output did not match the expected values."
        assert np.allclose(output_loss, expected_loss), "The cross-entropy loss output did not match the expected loss."