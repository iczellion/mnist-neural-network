import sys

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib
import matplotlib.pyplot as plt

class LinearLayer:
    """Class to represent a single Linear layer in a neural network.
       Performs a linear combination (weighted sum) of input parameters.

    Attributes:
        n_inputs: Number of input parameters.
        n_outputs: Number of output parameters.
        weights: Weights matrix for this layer.
        bias: Bias vector for this layer.
    """

    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        # Seed the random number generator for reproducibility
        np.random.seed(133)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = np.random.randn(n_outputs, n_inputs) * np.sqrt(2. / n_inputs) # Initialize weights with He initialization
        self.bias = np.zeros((n_outputs, 1)) # Initialize biases to zero
        self.inputs = None
        self.dweights = None
        self.dbias = None

    def forward(self, inputs) -> np.array:
        """Perform the forward pass for this layer."""
        self.inputs = inputs
        z1 = np.dot(self.weights, inputs) + self.bias
        return z1

    def backward(self, dvalues) -> np.array:
        """Perform the backward pass for this layer."""
        self.dweights = np.dot(dvalues, self.inputs.T)
        self.dbias = np.sum(dvalues, axis=1, keepdims=True)
        return np.dot(self.weights.T, dvalues)

class ActivationRelu:
    """Class to represent the ReLU activation function."""

    def __init__(self) -> None:
        self.inputs = {}

    def forward(self, inputs) -> np.array:
        """Perform the forward pass for ReLU activation."""
        self.inputs = inputs
        return np.maximum(inputs, 0)
    
    def backward(self, dvalues) -> np.array:
        """Perform the backward pass for ReLU activation."""
        return dvalues * (self.inputs > 0)

class ActivationSoftmaxLossCrossEntropy:
    """Class to combine Softmax activation and Cross-Entropy loss.
       
       Softmax activation is used in the output layer for multi-class classification.
       It converts raw scores (logits) into predicted probabilities.
    """

    def __init__(self) -> None:
        self.inputs = None
        self.output = None
        self.y_true = None
    
    def forward(self, inputs, y_true) -> float:
        """Perform the forward pass for Softmax activation and Cross-Entropy loss."""
        self.inputs = inputs
        self.y_true = y_true
        
        # Compute softmax probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        self.output = probabilities
        
        # Compute loss
        y_pred_clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
        correct_confidences = np.sum(y_true * np.log(y_pred_clipped), axis=0)
        loss = -np.mean(correct_confidences)
        
        return loss

    def backward(self) -> np.array:
        """Perform the backward pass for Softmax activation and Cross-Entropy loss."""
        samples = self.y_true.shape[1]
        dinputs = self.output - self.y_true
        return dinputs / samples

class NNValueEncoder:
    """Class to handle encoding of values for the neural network."""

    def __init__(self) -> None:
        pass

    def one_hot_encode(self, inputs) -> np.array:
        """One-hot encode the input values."""
        one_hot_output = np.zeros((inputs.size, inputs.max() + 1))
        one_hot_output[np.arange(inputs.size), inputs] = 1
        one_hot_output = one_hot_output.T
        return one_hot_output

class NeuralNetwork:
    """Class to represent the neural network model."""

    def __init__(self, layers, loss_function) -> None:
        self.layers = layers
        self.loss_function = loss_function

    def train(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array, epochs: int, learning_rate: float) -> None:
        """Train the neural network."""

        # Initialize one hot encoder for Y values
        enc = NNValueEncoder()
        y_one_hot_train = enc.one_hot_encode(y_train)

        for epoch in range(epochs):
            # Forward pass through all layers
            activation = x_train
            for layer in self.layers:
                activation = layer.forward(activation)
            
            # Calculate Softmax activation and cross-entropy loss
            loss = self.loss_function.forward(activation, y_one_hot_train)
            gradient = self.loss_function.backward()

            # Backward pass through all layers
            for layer in reversed(self.layers):
                gradient = layer.backward(gradient)
            
            # Update weights and biases
            for layer in self.layers:
                if isinstance(layer, LinearLayer):
                    layer.weights = layer.weights - learning_rate * layer.dweights
                    layer.bias = layer.bias - learning_rate * layer.dbias

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                pred = self.predict(x_test)
                accuracy = np.sum(pred == y_test) / y_test.size
                print(f"epoch: {epoch:>6} | loss: {loss:<20.8} | accuracy: {accuracy*100:.2f}%")
    
    def predict(self, x: np.array) -> np.array:
        """Predict the class labels for the given input data."""
        activation = x
        for layer in self.layers:
            activation = layer.forward(activation)
        
        class_labels = np.argmax(activation, axis=0, keepdims=True)
        return class_labels

def view_image(image_array: np.array) -> None:
    """Display the given image array."""
    current_image = image_array.reshape((28, 28)) * 255

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    # Set numpy print options for better readability
    np.set_printoptions(threshold=sys.maxsize)

    # Load MNIST training data into a numpy array
    data_train = pd.read_csv('./.tmp/mnist_train.csv')
    data_train = np.array(data_train)
    np.random.shuffle(data_train)

    # Load MNIST test data into a numpy array
    data_test = pd.read_csv('./.tmp/mnist_test.csv')  # Load MNIST dataset
    data_test = np.array(data_test)

    # Organize training data
    y_train = data_train[:, [0]].T
    x_train = data_train[:, 1:].T
    x_train = x_train / 255.

    # Organize test data
    y_test = data_test[:, [0]].T
    x_test = data_test[:, 1:].T
    x_test = x_test / 255.

    # Initialize loss function
    loss_func = ActivationSoftmaxLossCrossEntropy()

    # Initialize layers.
    # Our NN will have a simple two-layer architecture:
    #   Input layer a[0] will have 784 units corresponding to the 784 pixels in each 28x28 input image.
    #   A hidden layer a[1] will have 10 units with ReLU activation.
    #   Output layer a[2] will have 10 units corresponding to the ten digit classes with softmax activation.
    layers = [
        LinearLayer(x_train.shape[0], 10),
        ActivationRelu(),
        LinearLayer(10, 10)
    ]

    # Initialize neural network
    nn = NeuralNetwork(layers, loss_func)
    epochs = 400
    learning_rate = 0.10

    print("Starting training process...")
    nn.train(x_train, y_train, x_test, y_test, epochs, learning_rate)
    print("Done!")