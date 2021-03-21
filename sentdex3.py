#!/usr/bin/env python3

import numpy as np
from numpy.lib.arraypad import _slice_at_axis
from numpy.lib.function_base import _i0_2
import matplotlib.pyplot as plt

#rename inputs as capital X, per convention
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


def flattener(t):
    flat_list = [item for sublist in t for item in sublist]
    return flat_list

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)
## Rectified Linear Unit class


# neural network models are specified by biases and weights
# specify size of input (size of sample)
# specify number of neurons
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # init weights randomly
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # init biases to zero
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        # caluclating output for a given input
        self.output = np.dot(inputs, self.weights + self.biases)
## Layers class


# first size of layer1 is number of inputs from X
layer1 = Layer_Dense(4,5)
activation1 = Activation_ReLU()
# remember we're doing matrix mult, so we have to have 5 = 5
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)