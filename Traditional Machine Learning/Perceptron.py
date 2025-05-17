import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, in_size, random_state = None):
        if random_state is not None:
            np.random.seed(random_state)

        self.W = np.random.randn(in_size)
        self.b = np.random.randn()

        self.error_prog = []

    def fit(self, X, y, epochs, lr, verbose=False):
        for epoch in range(epochs):
            # forward pass
            z = self.W @ X.T + self.b
            y_pred = self.activation(z)

            # backpropagation
            dW, db = self.backprop(X, y, y_pred)

            # update weights
            self.W -= dW * lr
            self.b -= db * lr

            # compute error
            E = self.error(y, y_pred)
            self.error_prog.append(E)

            # print error
            if verbose:
                if (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1} | {E}")

    def predict(self, X):
        z = self.W @ X.T + self.b
        return np.round(self.activation(z))

    def backprop(self, X, y, y_pred):
        dL = -2 * (y - y_pred) * self.grad_activation(y_pred)
        dW = X.T @ dL
        db = np.sum(dL)
        return dW, db

    def activation(self, x):
        # sigmoid
        return 1 / (1 + np.exp(-x))

    def grad_activation(self, x):
        return self.activation(x) * (1 - self.activation(x))

    def error(self, y, y_pred):
        return np.sum(0.5 * (y - y_pred) ** 2)

    def plot_error(self):
        plt.plot(self.error_prog)
        plt.title("Prediction Error")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()