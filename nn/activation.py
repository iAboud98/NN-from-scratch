import numpy as np

def ReLU(inputs):
    return np.maximum(0, inputs)

def softmax(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims= True))
    return exp_values / np.sum(exp_values, axis = 1, keepdims= True)