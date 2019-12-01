import numpy as np
import pandas as pd

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
    
    def set_weights(self, weights):
        self.weights = weights

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
        activations = [[1]]
        for result in results:
            activations.append([self.activation_function(result[0])])

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
    
    def set_weights(self, weights):
        for i in range(len(self.neurons)):
            self.neurons[i].set_weights(weights[i])

class NeuralNetwork():
    def __init__(self):
        self.layers = []
    
    def train(self, x, y, shape=[], epochs=10):
        if len(shape) == 0:
            shape = [len(x[0]), len(x[0]), len(y[0])]

        # Initialize weights
        # Shape: (number of neurons, number of neurons in previous layer + 1)
        self.layers = []
        for i in range(len(shape) - 1):
            layer = Layer()
            layer.init_weights(shape[i], shape[i + 1])
            self.layers.append(layer)

        for _ in range(epochs):
            grad_table = []
            for i in range(len(self.layers)):
                grad_table.append(np.zeros(self.layers[i].get_dimension()))

            cost = 0
            for i in range(len(x)):
                row = x[i]
                target = y[i]
                activations = [np.transpose([np.concatenate(([1], row))])]
                for j in range(len(self.layers)):
                    activations.append(self.layers[j].feed_forward(activations[j]))
                
                activations[-1] = activations[-1][1:]
                cost = cost + sum([x[0]**2 for x in np.subtract(activations[-1], target)])

                activations_prime = np.subtract(np.ones((len(activations[-1]), 1)), activations[-1])
                delta = np.multiply(np.multiply(np.subtract(activations[-1], target), activations[-1]), activations_prime)
                #delta = np.subtract(activations[-1], target)

                for j in range(len(grad_table) - 1, -1, -1):
                    grad_table[j] = np.add(grad_table[j], np.matmul(delta, np.transpose(activations[j])))
                    activations_prime = np.subtract(np.ones((len(activations[j]), 1)), activations[j])
                    weight_times_delta = np.matmul(np.transpose(self.layers[j].get_weights()), delta)
                    delta = np.multiply(np.multiply(weight_times_delta, activations[j]), activations_prime)[1:]
            
            cost = cost / len(x)
            print (cost)

            # Regularization
            for l in range(len(grad_table)):
                for i in range(len(grad_table[l])):
                    for j in range(len(grad_table[l][i])):
                        if j == 0:
                            grad_table[l][i][j] = grad_table[l][i][j]/len(x)
                        else:
                            grad_table[l][i][j] = grad_table[l][i][j]/len(x) + self.layers[l].get_weights()[i][j]/len(x)

            for j in range(len(self.layers)):
                self.layers[j].set_weights(np.subtract(self.layers[j].get_weights(), grad_table[j]))

    def print_shape(self):
        for i in range(len(self.layers)):
            print ("Layer", i + 1)
            self.layers[i].print_shape()

train_data = pd.read_csv("train.csv")
prices = train_data["SalePrice"]
#print (np.median(prices), np.min(prices), np.max(prices), np.mean(prices))
cutoff_price = np.median(prices) # We will classify based on the price level
train_data = train_data[["LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "TotalBsmtSF", "SalePrice"]]
train_data = train_data.values
y = [[1 if t[-1] > cutoff_price else 0] for t in train_data]
x = [t[:-1] for t in train_data]

nn = NeuralNetwork()
nn.train(x, y)