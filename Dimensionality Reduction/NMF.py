import numpy as np

class NMF():
    def __init__(self, k, random_state=None):
        self.k = k

        if random_state is not None:
            np.random.seed(random_state)
    
    def init_values(self, X):
        self.X = X

        F = X.shape[0]
        T = X.shape[1]

        self.W = np.random.rand(F, self.k)
        self.H = np.random.rand(self.k, T)

        self.ones = np.ones((F,T))

    def update(self):
        self.W = self.W * (((self.X / (self.W @ self.H)) @ self.H.T) / (self.ones @ self.H.T))
        self.H = self.H * ((self.W.T @ (self.X / (self.W @ self.H))) / (self.W.T @ self.ones))

    def error(self):
        X_hat = self.W @ self.H
        return np.sum(self.X * np.log(self.X/X_hat) - self.X + X_hat)

    def fit(self, X, epochs, verbose=False):
        self.init_values(X)

        for epoch in range(epochs):
            self.update()
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1} | Error: {self.error()}")
        return self.W, self.H