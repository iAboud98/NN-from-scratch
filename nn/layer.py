import numpy as np
from nn.activation import ReLU, softmax

np.random.seed(0)

class Layer:

    def __init__(self, neurons_num, features_num):
        self.biases = np.zeros((1, neurons_num))
        self.weights = 0.10 * np.random.randn(neurons_num, features_num)

        # caches (needed for backward)
        self.inputs = None
        self.pre_activation = None

        # grads
        self.dweights = None
        self.dbiases = None
        self.dinputs = None

        # forward output
        self.output = None

    def forward(self, inputs):
        # cache inputs for backward
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights.T) + self.biases

    def run_activation(self):
        # cache pre-activation for ReLU backward
        self.pre_activation = self.output
        self.output = ReLU(self.output)

    def run_softmax(self):
        self.output = softmax(self.output)

    def backward(self, dvalues):
        # dW: (neurons, features)
        self.dweights = np.dot(dvalues.T, self.inputs)

        # db: (1, neurons)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # dX: (N, features)
        self.dinputs = np.dot(dvalues, self.weights)

    def backward_activation(self, dvalues):
        dZ = dvalues.copy()
        dZ[self.pre_activation <= 0] = 0
        return dZ

    def apply_gradients(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.biases  -= learning_rate * self.dbiases