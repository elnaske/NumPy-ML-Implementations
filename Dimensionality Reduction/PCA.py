import numpy as np

class PCA():
    def __init__(self, n_comp):
        self.n_comp = n_comp
        self.components = None
        self.mean = None

    def eigendecomposition(self, X, N):
        eigenvalues = []
        eigenvectors = []

        for _ in range(N):
            # Get eigenvector and eigenvalue through power iteration
            eigenvalue, eigenvector = self.power_iteration(X)
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)

            # Deflate the matrix
            X = self.deflate(X, eigenvector, eigenvalue)

        eigenvalues = np.stack(eigenvalues, axis = 0)
        eigenvectors = np.stack(eigenvectors, axis = 1)

        return eigenvalues, eigenvectors
    
    def power_iteration(self, X, k=100):
        y = np.random.rand(X.shape[1])

        for _ in range(k):
            y = (X @ y) / np.linalg.norm(X @ y)

        eigenvalue = (y @ (X @ y)) / (y @ y)
        return eigenvalue, y
    
    def deflate(self, X, v, l):
        return X - l * np.outer(v, v)
    
    def fit(self, X):
        # Center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Covariance matrix
        covmat = np.cov(X_centered, rowvar=False)

        # Get top k eigenvectors
        _, eigvecs = self.eigendecomposition(covmat, self.n_comp)

        self.components = eigvecs

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class KernelPCA(PCA):
    def __init__(self, n_comp, sigma=1.0):
        self.n_comp = n_comp
        self.sigma = sigma
        self.components = None
        self.mean = None

    def rbf_kernel(self, X, Y=None):
        if Y is None:
            Y = X
        # Compute square euclidean distances
        diffs = Y[:,np.newaxis,:] - X[np.newaxis,:,:]
        sq_dists = np.sum(diffs**2, axis=-1)

        return np.exp(-sq_dists / self.sigma**2)
    
    def fit(self, X):
        self.X_fit = X

        # Get the RBF kernel
        K = self.rbf_kernel(X)

        # Get top k eigenvectors
        _, eigvecs = self.eigendecomposition(K, self.n_comp)

        self.components = eigvecs

    def transform(self, X):
        K = self.rbf_kernel(X, self.X_fit)
        return K.T @ self.components