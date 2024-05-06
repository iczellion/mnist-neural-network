import sys

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt

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
        np.random.seed(133)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = np.random.rand( n_outputs, n_inputs ) - 0.5
        self.bias = np.random.rand( n_outputs, 1 ) - 0.5
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
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        self.output = probabilities
        return probabilities

    """
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
    """
    def backward(self, dvalues):
        print(f"DVALUES: {dvalues}")
        return dvalues

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
        In the context of calculating cross-entropy loss using probabilities, it's essential to ensure that the probabilities are within a valid range (i.e., between 0 and 1).
        The reason for this is that the cross-entropy formula involves taking the logarithm of the predicted probabilities. If any predicted probability is exactly 0 or 1,
        taking the logarithm of it would result in negative infinity or undefined values, respectively.
        """

        # The np.clip() function is used to limit the values in an array to be within a specified range.
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

        cross_entropy_loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

        return cross_entropy_loss

    def backward(self, y_true, y_pred):
        samples = len(y_pred)
        gradient = y_pred - y_true
        gradient = gradient / samples  # Normalize the gradient by the number of samples
        return gradient

class NeuralNetwork:

    def __init__(self, layers, loss_function) -> None:
        self.layers = layers
        self.loss_function = loss_function

    def train(self, x_train, y_train, x_test, y_test, epochs, learning_rate):

        # Initialize one hot encoder for Y values
        enc = NNValueEncoder()
        y_one_hot_train = enc.one_hot_encode(y_train)

        for epoch in range(epochs):
            # Forward pass
            activation = x_train
            for layer in self.layers:
                activation = layer.forward(activation)
            
            # Loss calculation
            loss = self.loss_function.calculate(y_one_hot_train, activation)

            # Backward pass
            gradient = self.loss_function.backward(y_one_hot_train, activation)
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient)
            
            # Update weights and biases
            for layer in self.layers:
                if isinstance(layer, LinearLayer):
                    layer.weights = layer.weights - learning_rate * layer.dweights
                    layer.bias = layer.bias - learning_rate * layer.dbias

            if epoch % 10 == 0:
                pred = self.predict(x_test)
                accuracy = np.sum(pred == y_test) / y_test.size
                print(f"Test: {y_test[0, 1:10]} | Pred: {pred[0, 1:10]}")
                print(f"Epoch {epoch} | loss: {loss}, accuracy: {accuracy*100}%")
    
    def predict(self, x):
        activation = x
        for layer in self.layers:
            activation = layer.forward(activation)
        
        class_labels = np.argmax(activation, axis=0, keepdims=True)
        return class_labels

def view_image(image_array):
    current_image = image_array.reshape((28, 28)) * 255

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

if __name__ == "__main__":

    # Load mnist training data in a numpy array
    data_train = pd.read_csv('./.tmp/mnist_train.csv')
    data_train = np.array(data_train)
    np.random.shuffle(data_train)

    # Load mnist test data in a numpy array
    data_test = pd.read_csv('./.tmp/mnist_test.csv') # load mnist dataset
    data_test = np.array(data_test)

    # Organize training data
    y_train = data_train[:, [0]].T
    x_train = data_train[:,1:].T
    x_train = x_train / 255.

    # Organize test data
    y_test = data_test[:, [0]].T
    x_test = data_test[:,1:].T
    x_test = x_test / 255.

    # Initial loss function
    loss_func = CrossentropyLoss()

    # Initialize layers.
    # Our NN will have a simple two-layer architecture, where:
    #   Input layer a[0] will have 784 units corresponding to the 784 pixels in each 28x28 input image.
    #   A hidden layer a[1] will have 10 units with ReLU activation
    #   Output layer a[2] will have 10 units corresponding to the ten digit classes with softmax activation.
    layers = [
        LinearLayer(x_train.shape[0], 10),
        ActivationRelu(),
        LinearLayer(10, 10),
        ActivationSoftmax()
    ]

    # Initialize neural network
    nn = NeuralNetwork(layers, loss_func)
    epochs = 400
    learning_rate = 0.10

    print("Starting training process...")
    nn.train(x_train, y_train, x_test, y_test, epochs, learning_rate)

    #print(y_one_hot_train.shape)
    np.set_printoptions(threshold=sys.maxsize)

    """
    print(f"PREDICT: {nn.predict(x_test[:, [7]])}")
    view_image(x_test[:, 7])

    print(f"PREDICT: {nn.predict(x_test[:, [18]])}")
    view_image(x_test[:, 18])

    print(f"PREDICT: {nn.predict(x_test[:, [19]])}")
    view_image(x_test[:, 19])
    """