#!/usr/bin/env python3


import numpy as np

#first hidden layer

inputs = [  [1, 2, 3, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8], ]

weights = [ [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87], ]

biases = [  2,\
            3,\
            0.5 ]

# make a second layer, again same kludge, then doing
# objects later

weights2 = [ [0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13], ]

biases2 = [ 1,\
            2,\
            -0.5 ]



layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
print(layer1_outputs)
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)
