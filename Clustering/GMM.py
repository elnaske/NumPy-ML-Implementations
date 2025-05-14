import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel():
    def __init__(self, X, k):
        self.X = X
        self.k = k
        self.N = X.shape[0]

        self.P = np.ones(k) / k # equal chance for both clusters
        self.means = X[np.random.choice(self.N, k, replace=False)] # randomly chosen means like for k-means
        self.cov = np.array([np.cov(X, rowvar=False) for _ in range(k)])
        self.U = np.zeros((self.N, self.k))
        
    def e_step(self):
        for j in range(self.k):
            self.U[:, j] = self.P[j] * multivariate_normal.pdf(self.X, self.means[j], self.cov[j])

        self.U /= np.sum(self.U, axis=1, keepdims=True)

    def m_step(self):
        sum_U = np.sum(self.U, axis=0)

        self.P = sum_U / self.N
        self.means = self.U.T @ self.X / sum_U[:, np.newaxis]

        for j in range(self.k):
            dist = np.squeeze(self.X - self.means[j])
            self.cov[j] = np.sum(self.U[:, j] * dist @ dist.T) / sum_U[j]

    def fit(self, epochs):
        for _ in range(epochs):
            self.e_step()
            self.m_step()

    def predict(self):
        return np.argmax(self.U, axis=1)