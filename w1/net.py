# Referenced from Michael Nielsen's online book Chapter 1

import numpy as np
import random

class Net(object):

    def __init__(self, size):
        self.size = size
        self.layers = len(size)
        self.bias = [np.random.randn(x, 1) for x in size[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(size[:-1], size[1:])]

    def sgd(self, epochs, batch_size, lr, train, test=None):
        n = len(train)
        for i in range(epochs):
            #random.shuffle(train)
            mini_batch = [train[b:b+batch_size] for b in range(0, n, batch_size)]
            for batch in mini_batch:
                self.update(batch, lr)
            print("Epoch {0} complete".format(i))

    def update(self, mini_batch, rate):
        n = len(mini_batch)
        for b in self.bias:
            gradient_b = [np.zeros(b.shape)]
        for w in self.weights:
            gradient_w = [np.zeros(w.shape)]

        for X, y in mini_batch:
            delta_b, delta_w = self.backprop(X, y)
            gradient_b = [a+b for a,b in zip(gradient_b, delta_b)]
            gradient_w = [a+b for a,b in zip(gradient_w, delta_w)]

        #Update rule
        for w, dw in zip(self.weights, gradient_w):
            self.weights = [w - (rate/n) * dw]
        
        for b, db in zip(self.bias, gradient_b):
            self.bias = [b - (rate/n) * db]

    def backprop(self, x, y):
        temp = x
        activations = [x]
        combination = []

        gradient_b = [np.zeros(b.shape) for b in self.bias]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

    ## Feedforward pass
        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, temp) + b
            combination.append(z)
            activations.append(sigmoid(z))

        ## Backward pass
        delta = (activations[-1] - y) * sigmoid_d(combination[-1])
        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, activations[-2].transpose())

        for n in range(2, self.layers):
            linear_comb = combination[-n]
            derivative = sigmoid_d(linear_comb)
            delta = np.dot(self.weights[-n+1].transpose(), delta) * derivative
            gradient_b[-n] = delta
            gradient_w[-n] = np.dot(delta, activations[-n-1].transpose())

        return (gradient_b, gradient_w)

    def ff(self, a):
        """Makes a forward pass through the network for the input """
        for w, b in zip(self.weights, self.bias):
            layer_i = (np.dot(w, layer_i)+b)
            a = sigmoid(layer_i)       

        return a

def sigmoid(z):
    """Outputs the sigmoid of the input"""
    s = 1. / (1. + np.exp(-z))
    return s

def sigmoid_d(z):
    return sigmoid(z) * (1 - sigmoid(z))
