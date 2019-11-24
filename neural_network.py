import numpy as np

class Neuron():
    def __init__(self):
        self.weights = []
    
    def init_weights(self, dimension):
        # +1 for bias term
        self.weights = np.random.normal(0.0, 1.0, dimension + 1)

    def print_shape(self):
        print (self.weights)

    def get_weights(self):
        return self.weights

class Layer():
    def __init__(self):
        self.neurons = []

    def init_weights(self, prev_dimension, dimension):
        self.neurons = []
        for i in range(dimension):
            neuron = Neuron()
            neuron.init_weights(prev_dimension)
            self.neurons.append(neuron)
    
    def print_shape(self):
        for neuron in self.neurons:
            neuron.print_shape()
    
    def feed_forward(self, values):
        # Vectorize
        matrix = [x.get_weights() for x in self.neurons]
        values = [1] + values
        results = np.matmul(matrix, values)
        activations = []
        for result in results:
            activations.append(np.tanh(result))

        return activations

class NeuralNetwork():
    def __init__(self):
        self.layers = []
    
    def train(self, x, y, shape=[]):
        if len(shape) == 0:
            shape = [len(x[0]), len(x[0]), len(y[0])]

        # Initialize weights
        # Shape: (number of neurons, number of neurons in previous layer + 1)
        self.layers = []
        for i in range(len(shape) - 1):
            layer = Layer()
            layer.init_weights(shape[i], shape[i + 1])
            self.layers.append(layer)

        for row in x:
            activations = self.layers[0].feed_forward(row)
            for i in range(len(self.layers) - 1):
                activations = self.layers[i + 1].feed_forward(activations)
            
            print(activations)
    
    def print_shape(self):
        for i in range(len(self.layers)):
            print ("Layer", i + 1)
            self.layers[i].print_shape()

nn = NeuralNetwork()
nn.train([[3, 5, 3]], [[1]])