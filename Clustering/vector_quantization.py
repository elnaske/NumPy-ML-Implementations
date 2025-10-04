import numpy as np

class ResidualVQ():
    def __init__(self, n_levels, n_codebooks):
        self.n_levels = n_levels
        self.n_codebooks = n_codebooks

    def fit(self, X):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        self.codebook = np.zeros((self.n_levels, self.n_features, self.n_codebooks))

        residuals = np.copy(X)

        for i in range(self.n_codebooks):
            # k-means clustering
            codes = self._cluster(residuals)
            self.codebook[:,:,i] += codes

            # map residuals to codes
            preds = self._map(residuals, i)
            quantized = self._quantize(preds, i)

            # update residuals
            residuals -= quantized

    def transform(self, X):
        encoded = np.zeros((X.shape[0], self.n_codebooks), dtype=int)
        residuals = np.copy(X)

        for i in range(self.n_codebooks):
            preds = self._map(residuals, i)
            quantized = self._quantize(preds, i)

            encoded[:,i] += preds

            residuals -= quantized
        
        return encoded

    def decode(self, enc):
        X_hat = 0
        for i in range(self.n_codebooks):
            preds = enc[:,i]
            X_hat += self._quantize(preds, i)
        return X_hat

    def _cluster(self, X, n_iter=100):
        centroids = X[np.random.choice(X.shape[0], size=self.n_levels, replace=False)]

        for _ in range(n_iter):
            dist = self._distance(X, centroids)
            preds = np.argmin(dist, axis = -1)
            centroids = np.array([np.mean(X[preds == i], axis = 0) for i in range(self.n_levels)])
        return centroids

    def _map(self, X, idx):
        # return the idxs of the closest code to X in codebook idx
        dist = self._distance(X, self.codebook[:,:,idx])
        preds = np.argmin(dist, axis = -1)
        return preds

    def _quantize(self, preds, idx):
        return np.array([self.codebook[j,:,idx] for j in preds])

    def _distance(self, a, b):
        # Euclidean distance
        return np.sqrt(np.sum((a[:, np.newaxis] - b) ** 2, axis=-1))