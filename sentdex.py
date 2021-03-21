#!/usr/bin/env python3

# working on the neural networks from scratch series by sentdex
# uses python ~3 and numpy to build neural networks and implement
# forward and back prop, and the various supporting elements to
# implement it all.

# https://www.youtube.com/watch?v=Wo5dMEP_BbI
# nnfs.io

import sys
import numpy as np
import matplotlib


# make a single neuron, 3 inputs + bias

inputs = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

# output = inputs.weights + bias
#output = [0,0,0]
#output = []
output = [None] * 3
output[0] =    inputs[0]*weights1[0]+\
            inputs[1]*weights1[1]+\
            inputs[2]*weights1[2]+\
            inputs[3]*weights1[3]+\
            bias1
output[1] =    inputs[0]*weights2[0]+\
            inputs[1]*weights2[1]+\
            inputs[2]*weights2[2]+\
            inputs[3]*weights2[3]+\
            bias2
output[2] =    inputs[0]*weights2[0]+\
            inputs[1]*weights2[1]+\
            inputs[2]*weights2[2]+\
            inputs[3]*weights2[3]+\
            bias3




print(output)