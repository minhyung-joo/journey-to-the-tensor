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

    def get_dimension(self):
        return len(self.weights)

class Layer():
    def __init__(self):
        self.neurons = []
        self.activation = "sigmoid"

    def init_weights(self, prev_dimension, dimension):
        self.neurons = []
        for i in range(dimension):
            neuron = Neuron()
            neuron.init_weights(prev_dimension)
            self.neurons.append(neuron)

    def set_activation(self, activation):
        self.activation = activation
    
    def print_shape(self):
        for neuron in self.neurons:
            neuron.print_shape()
    
    def feed_forward(self, values):
        # Vectorize
        matrix = [x.get_weights() for x in self.neurons]
        results = np.matmul(matrix, values)
        activations = [1]
        for result in results:
            activations.append(self.activation_function(result))

        return activations

    def activation_function(self, value):
        if self.activation == "tanh":
            return np.tanh(value)
        elif self.activation == "relu":
            return np.maximum(0, value)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-value))
    
    def get_dimension(self):
        return (len(self.neurons), self.neurons[0].get_dimension())

    def get_weights(self):
        return [x.get_weights() for x in self.neurons]

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

        grad_table = []
        for i in range(len(self.layers)):
            grad_table.append(np.zeros(self.layers[i].get_dimension()))

        for i in range(len(x)):
            row = x[i]
            target = y[i]
            activations = [[1] + row]
            for j in range(len(self.layers)):
                activations.append(self.layers[j].feed_forward(activations[j]))
            
            activations[-1] = activations[-1][1:]
            
            diff = 0
            for j in range(len(activations[-1])):
                diff = diff + (activations[-1][j] - target[j])**2
            loss = diff / 2

            delta = np.subtract(activations[-1], target)
            for k in range(len(grad_table[-1])):
                delta[k] = delta[k] * activations[-1][k] * (1 - activations[-1][k])

            for j in range(len(grad_table) - 1, -1, -1):
                newDelta = []
                for k in range(len(grad_table[j])):
                    for l in range(len(self.layers[j].get_weights()[k])):
                        grad_table[j][k][l] = grad_table[j][k][l] + delta[k] * self.layers[j].get_weights()[k][l]
                    for l in range(len(activations[j])):
                        newDelta.append(delta[k]*activations[j][l]*(1 - activations[j][l]))
                delta = newDelta[1:]
            
            print (grad_table)

    def print_shape(self):
        for i in range(len(self.layers)):
            print ("Layer", i + 1)
            self.layers[i].print_shape()

nn = NeuralNetwork()
nn.train([[3, 5, 3]], [[1]])