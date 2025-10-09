# this class is not used in NN, its only to show what happens inside a single neuron

import numpy as np

np.random.seed(0)

class Neuron:

    def __init__(self, features_num):
        self.weights = np.random.randn(1, features_num)
        self.bias = 0.0
    
    def forward(self, input):
        self.output = np.dot(input, self.weights.T) + self.bias

n1 = Neuron(3)
n1.forward([[1,2,3],[2,2,4]])
print(n1.output)