import numpy as np

class KMeans():
    def __init__(self, X, k):
        self.X = X
        self.k = k
        self.centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    def predict(self):
        distances = abs(self.X[:,np.newaxis] - self.centroids)
        return np.argmin(distances, axis = 1)

    def update_centroids(self, predictions):
        return np.array([np.mean(self.X[predictions == i]) for i in range(self.k)])

    def fit(self, epochs):
        for _ in range(epochs):
            predictions = self.predict()
            self.centroids = self.update_centroids(predictions)