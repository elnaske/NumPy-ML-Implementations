import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, in_size, random_state=None):
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


class MultinomialPerceptron(Perceptron):
    def __init__(self, in_size, out_size, random_state = None):
        if random_state is not None:
            np.random.seed(random_state)

        self.W = np.random.randn(out_size, in_size)
        self.b = np.random.randn()

        self.error_prog = []
    
    def fit(self, X, y, epochs, lr, verbose=False):
        # One-hot encode labels
        y_onehot = self.one_hot(y, self.W.shape[0])
        
        for epoch in range(epochs):
            # Forward pass
            z = X @ self.W.T + self.b
            y_pred = self.activation(z)

            # Backpropagation
            dW, db = self.backprop(X, y_onehot, y_pred)

            # Update weights
            self.W -= lr * dW
            self.b -= lr * db

            # Compute error
            E = self.error(y_onehot, y_pred)
            self.error_prog.append(E)

            # Print error
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1} | Error: {E:.4f}")
    
    def predict(self, X):
        z = X @ self.W.T + self.b
        probs = self.activation(z)
        return np.argmax(probs, axis=1)

    def backprop(self, X, y, y_pred):
        dL = (y_pred - y) / X.shape[0]
        dW = dL.T @ X
        db = np.sum(dL, axis=0)
        return dW, db

    def activation(self, x):
        # Softmax (subtracting max for numerical stability)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def error(self, y, y_pred):
        # Clip logits for numerical stability
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return - np.sum(y * np.log(y_pred))

    def one_hot(self, y, num_classes):
        return np.eye(num_classes)[y]