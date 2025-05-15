import numpy as np
from scipy.stats import mode

class KNN():
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # Compute distances
        diff = np.abs(self.X_train[np.newaxis,:] - X[:,np.newaxis])
        dist = np.sum(diff, axis=2)

        # Finding indices of the closest data points
        np.random.seed(42)
        sorted_idxs = np.argsort(dist, axis=-1)

        # Only keep k nearest elements
        k_nearest = sorted_idxs[:,:self.k]

        # Get the labels
        labels = np.repeat(np.expand_dims(self.y_train, axis=1), k_nearest.shape[0], axis=1).T
        kN_labels = np.stack([labels[i, k_nearest[i,:]] for i in range(labels.shape[0])])

        # Take the mode for prediction
        predictions = mode(kN_labels, axis = 1).mode

        return predictions

    def accuracy(self, y, y_hat):
        return np.sum(y == y_hat) / y.shape[0]

    def evaluate(self, X_test, y_test, k):
        y_hat = self.predict(X_test, k)
        acc = self.accuracy(np.squeeze(y_test), y_hat)
        return acc