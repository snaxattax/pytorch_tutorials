#!/usr/bin/env python3

import math
import numpy as np

layer_outputs = [   [4.8, 1.21, 2.385],
                    [8.9, -1.81, 0.2],
                    [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)

print(exp_values)

'''
#sum of columns
print(np.sum(layer_outputs, axis=0))

#sum of rows
print(np.sum(layer_outputs, axis=1))
'''

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)