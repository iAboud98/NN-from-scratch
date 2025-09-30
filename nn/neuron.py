class Neuron:

    def __init__(self, inputs, weights, bias):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias

    def linear_output(self):
        counter = 0
        for _ in range(len(self.inputs)):
            counter += self.inputs[_] * self.weights[_]
        return counter + self.bias