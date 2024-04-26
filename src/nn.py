import pandas as pd
import numpy as np

class LinearLayer:
    """Class to represent a single Linear layer in a neural network.
       We perform what is called a "linear combination" or "weighted sum" operation

    Attributes:
        n_inputs: Number of parameters as input.
        n_outputs: Number of outputs for this layer.
        weights: Weights for this layer.
        bias: Bias for this layer.
    """

    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = np.random.randn ( n_outputs, n_inputs ) * 0.01
        self.bias = np.zeros( (n_outputs, 1) )
        self.inputs = None
        self.dweights = None
        self.dbias = None
    
    def forward(self, inputs):
        self.inputs = inputs
        z1 = np.dot(self.weights, inputs) + self.bias
        return z1

    def backward(self, dvalues):
        self.dweights = np.dot(dvalues, self.inputs.T)
        self.dbias = np.sum(dvalues, axis=1, keepdims=True)
        return np.dot(self.weights.T, dvalues)

class ActivationRelu:

    def __init__(self) -> None:
        self.inputs = {}

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(inputs, 0)
    
    def backward(self, dvalues):
        return dvalues * (self.inputs > 0)

class ActivationSoftmax:
    """Class to represent a Softmax activation in a neural network.

       The softmax activation function is commonly used in the output layer of a neural network when dealing with multi-class classification problems.
       It converts raw scores or logits into predicted probabilities of each class for each input.
    """

    def __init__(self) -> None:
        self.inputs = None
        self.output = None
    
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        self.output = probabilities
        return probabilities

    def backward(self, dvalues):
        # Initialize gradient array
        dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output.T, dvalues.T)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            dinputs[:, index] = np.dot(jacobian_matrix, single_dvalues)

        return dinputs

class NNValueEncoder:

    def __init__(self) -> None:
        pass

    def one_hot_encode(self, inputs):
        one_hot_output = np.zeros((inputs.size, inputs.max() + 1))
        one_hot_output[np.arange(inputs.size), inputs] = 1
        one_hot_output = one_hot_output.T
        return one_hot_output

class CrossentropyLoss:

    def calculate(self, y_true, y_pred):

        """
        The np.clip() function is used to limit the values in an array to be within a specified range.
        Clipping values means that any values in the array that fall outside the specified range are set to the nearest value within that range.

        In the context of calculating cross-entropy loss using probabilities, it's essential to ensure that the probabilities are within a valid range (i.e., between 0 and 1).
        The reason for this is that the cross-entropy formula involves taking the logarithm of the predicted probabilities. If any predicted probability is exactly 0 or 1,
        taking the logarithm of it would result in negative infinity or undefined values, respectively.
        """
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

        cross_entropy_loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

        return cross_entropy_loss

    def backward(self, y_true, y_pred):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        gradient = -y_true / y_pred_clipped
        gradient = gradient / samples
        return gradient

if __name__ == "__main__":

    data = pd.read_csv('./.tmp/mnist_test.csv') # load mnist dataset

    data = np.array(data)
    m, n = data.shape
    #np.random.shuffle(data)  # shuffle before splitting into dev and training sets

    data_dev = data[0:1000]
    Y_dev = data_dev[:, 0].T
    X_dev = data_dev[:,1:].T
    X_dev = X_dev / 255.

    # One hot encode Y values
    enc = NNValueEncoder()
    one_hot_Y = enc.one_hot_encode(Y_dev)

    # Initialize layers.
    # Our NN will have a simple two-layer architecture, where:
    #   Input layer a[0] will have 784 units corresponding to the 784 pixels in each 28x28 input image.
    #   A hidden layer a[1] will have 10 units with ReLU activation
    #   Output layer a[2] will have 10 units corresponding to the ten digit classes with softmax activation.
    layer_1 = LinearLayer(X_dev.shape[0], 10)
    act_1 = ActivationRelu()
    layer_2 = LinearLayer(10, 10)
    act_2 = ActivationSoftmax()
 
    # Forward pass
    out1 = layer_1.forward(X_dev)
    out_act1 = act_1.forward(out1)
    out2 = layer_2.forward(out_act1)
    out_act2 = act_2.forward(out2)

    # Calculate loss
    loss = CrossentropyLoss()
    print("Loss:", loss.calculate(Y_dev, out_act2))

    #  Backward pass
    dloss = loss.backward(one_hot_Y, out_act2)
    dout2 = act_2.backward(dloss)
    dout1 = layer_2.backward(dout2)
    dout_act1 = act_1.backward(dout1)
    dX = layer_1.backward(dout_act1)