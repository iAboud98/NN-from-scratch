import numpy as np

class BackPropagation:

    def __init__(self, predictions, true_vals):
        self.predictions = predictions
        self.true_vals = true_vals


    def one_hot(self, classes_number):
        self.true_vals = self.true_vals.flatten().astype(int)
        encodings = np.zeros((len(self.true_vals), classes_number))
        encodings[np.arange(len(self.true_vals)), self.true_vals] = 1

        return encodings

    def cross_entropy(self, encoded_true):
        log_preds = np.log(self.predictions)
        multiplied = encoded_true * log_preds

        return -np.sum(multiplied, axis=1)

