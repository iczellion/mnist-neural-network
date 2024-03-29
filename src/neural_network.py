import numpy as np

class ShallowNeuralNetwork:

    def __init__(self, n_inputs: int, n_neurons: int, n_outputs: int, learning_rate: int = 0.001) -> None:
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights_l1 = np.random.randn ( n_neurons, n_inputs ) * 0.01
        self.weights_l2 = np.random.randn ( n_outputs, n_neurons ) * 0.01
        self.bias_l1 = np.zeros( (n_neurons, 1) )
        self.bias_l2 = np.zeros( (n_outputs, 1) )
        self.n_hidden_layers = 1
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
    
    def __tanh(self, x):
        tanh = ( np.exp(x) - np.exp(-x) ) / ( np.exp(x) + np.exp(-x) )
        return tanh

    def foward_propagation(self, inputs):
        z1 = np.add( np.dot(self.weights_l1, inputs), self.bias_l1)
        a1 = self.__tanh(z1)
        z2 = np.add( np.dot(self.weights_l2, a1), self.bias_l2)
        a2 = self.__tanh(z2)
        return a2
    
    def back_propagation(self):
        pass


if __name__ == "__main__":
    nn = ShallowNeuralNetwork(3, 4, 1)
    a = np.matrix([1, 3, 4]).T

    print("Doing a forward pass:")
    z = nn.foward_propagation(a)