import numpy as np

class LinearLayer:
    """Class to represent a single Linear layer in a neural network.

    Attributes:
        n_inputs: Number of parameters as input.
        n_outputs: Number of outputs for this layer.
        weights: Weights for this layer.
        bias: Bias for this layer.
    """

    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.cache = {}

        self.weights = np.random.randn ( n_outputs, n_inputs ) * 0.01
        self.bias = np.zeros( (n_outputs, 1) )
    
    def forward(self, inputs):
        self.cache = inputs
        z1 = np.dot(self.weights, inputs) + self.bias
        return z1

class ActivationRelu:

    def __init__(self) -> None:
        self.cache = {}

    def forward(self, inputs):
        self.cache = inputs
        return np.maximum(inputs, 0)
    
    def backward(self, dvalues):
        return dvalues > 0

class ActivationSoftmax:

    def __init__(self) -> None:
        self.cache = {}
    
    def forward(self, inputs):
        self.cache = inputs
        return np.exp(inputs) / sum(np.exp(inputs))

class CrossentropyLoss:

    def calculate(self, y_true, y_pred):

        """
        The np.clip() function is used to limit the values in an array to be within a specified range.
        Clipping values means that any values in the array that fall outside the specified range are set to the nearest value within that range.

        In the context of calculating cross-entropy loss using probabilities, it's essential to ensure that the probabilities are within a valid range (i.e., between 0 and 1).
        The reason for this is that the cross-entropy formula involves taking the logarithm of the predicted probabilities. If any predicted probability is exactly 0 or 1,
        taking the logarithm of it would result in negative infinity or undefined values, respectively.
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        cross_entropy_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        #cross_entropy_loss = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)

        return cross_entropy_loss


if __name__ == "__main__":
    x = np.array(
        [
            [0, 1, 1],
            [2, 3, 5],
            [8, 13, 21],
            [34, 55, 89],
            [144, 233, 377],
            [610, 987, 1597],
            [2584, 4181, 6765],
            [10946, 17711, 28657],
            [46368, 75025, 121393]
        ]
    ).T
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).T

    layer_1 = LinearLayer(3, 4)
    act_1 = ActivationRelu()
    layer_2 = LinearLayer(4, 1)
    act_2 = ActivationSoftmax()

    # Do a forward pass
    out1 = layer_1.forward(x)
    out_act1 = act_1.forward(out1)
    out2 = layer_2.forward(out_act1)
    out_act2 = act_2.forward(out2)

    # Calculate loss
    loss = CrossentropyLoss()
    dz1 = loss.calculate(y, out_act2)

    #print(out_act1)
    
    #der = act_2.backward(out2)
    print(x.shape)