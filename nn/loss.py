import numpy as np

class BackPropagation:

    def __init__(self, predictions, true_vals):
        self.predictions = predictions
        self.true_vals = true_vals
        self.encoded_true = None

    def one_hot(self, classes_number):
        self.true_vals = self.true_vals.flatten().astype(int)
        encodings = np.zeros((len(self.true_vals), classes_number))
        encodings[np.arange(len(self.true_vals)), self.true_vals] = 1
        return encodings

    def cross_entropy(self, encoded_true):
        self.encoded_true = encoded_true

        eps = 1e-12
        clipped_preds = np.clip(self.predictions, eps, 1.0 - eps)

        log_preds = np.log(clipped_preds)
        multiplied = encoded_true * log_preds

        losses = -np.sum(multiplied, axis=1)
        return losses.mean()

    def d_softmax_cross_entropy(self):
        N = self.predictions.shape[0]
        return (self.predictions - self.encoded_true) / N