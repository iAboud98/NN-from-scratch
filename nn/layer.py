import numpy as np
from activation import ReLU

np.random.seed(0)

class Layer:

    def __init__(self, features_num, neurons_num):
        self.biases = np.zeros((1,neurons_num))
        self.weights = 0.10 * np.random.randn(neurons_num, features_num)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights.T) + self.biases

    def run_activation(self, inputs):
        self.output = ReLU(inputs)

l1 = Layer(2, 5)
l1.forward([[1,2], [4,2], [3,1]])
l1.run_activation(l1.output)