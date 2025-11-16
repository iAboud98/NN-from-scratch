import numpy as np
from nn.activation import ReLU, softmax

np.random.seed(0)

class Layer:

    def __init__(self, neurons_num, features_num):
        self.biases = np.zeros((1,neurons_num))
        self.weights = 0.10 * np.random.randn(neurons_num, features_num)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights.T) + self.biases

    def run_activation(self, inputs):
        self.output = ReLU(inputs)

    def run_softmax(self, inputs):
        self.output = softmax(inputs)