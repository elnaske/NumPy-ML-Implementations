import numpy as np

class KMeans():
    def __init__(self, k):
        self.k = k

    def predict(self, X):
        if len(X.shape) == 1:
            X = np.expand_dims(X, 1)

        distances = self._distance(X, self.centroids)
        return np.argmin(distances, axis = -1)

    def update_centroids(self, X, predictions):
        return np.array([np.mean(X[predictions == i], axis = 0) for i in range(self.k)])

    def fit(self, X, epochs):
        if len(X.shape) == 1:
            X = np.expand_dims(X, 1)

        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for _ in range(epochs):
            predictions = self.predict(X)
            self.centroids = self.update_centroids(X, predictions)
    
    def _distance(self, a, b):
        # Euclidean distance
        return np.sqrt(np.sum((a[:, np.newaxis] - b) ** 2, axis=-1))