import numpy as np
from Perceptron import Perceptron

class AdaPerceptron(Perceptron):
    def __init__(self, in_size, w_m):
        self.W = np.random.randn(in_size)
        self.b = np.random.randn()

        self.w_m = w_m

        self.error_prog = []
    
    def predict(self, X):
        z = self.W @ X.T + self.b
        return self.activation(z)

    def backprop(self, X, y, y_pred):
        dL = -2 * self.w_m * (y - y_pred) * self.grad_activation(y_pred)
        dW = X.T @ dL
        db = np.sum(dL)
        return dW, db

    def activation(self, x):
        # tanh
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    def grad_activation(self, x):
        return 1 - self.activation(x) ** 2

    def error(self, y, y_pred):
        return self.w_m @ (y - y_pred) ** 2

class AdaBoost():
    def __init__(self, X, y, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)

        self.X = X
        self.y = y

        self.T = X.shape[0] # Number of samples
        self.F = X.shape[1] # Number of features

        # Initialize sample weights
        self.w_0 = np.random.rand(self.T) * 1e-6

        self.sample_weights = [self.w_0]
        self.learner_weights = []
        self.perceptron_weights = []

    def fit(self, rounds, wl_epochs, wl_lr):
        for m in range(rounds):
            # Get the sample weights from the previous round
            w_m = self.sample_weights[-1]

            # Train perceptron and get the weights and predictions
            phi_m, phi_m_x = self.compute_phi(wl_epochs, wl_lr, w_m)

            # Calculate learner weight
            beta_m = self.compute_beta(w_m, phi_m_x)

            # Update sample weights
            w_tm_1 = self.compute_w(w_m, phi_m_x, beta_m)

            if m < rounds:
                # Skip if this is the final round
                self.sample_weights.append(w_tm_1)
            self.learner_weights.append(beta_m)
            self.perceptron_weights.append(phi_m)

        self.w = np.stack(self.sample_weights)
        self.beta = np.array(self.learner_weights)
        self.phi = np.stack(self.perceptron_weights)

    def predict(self, X):
        W = self.phi[:, :2]
        b = self.phi[:, 2]
        b = np.repeat(np.expand_dims(b, axis=1), X.shape[0], axis=1)
        z = W @ X.T + b
        phi_x = np.sign(self.tanh(z))
        return np.sign(self.beta @ phi_x)

    def compute_phi(self, epochs, lr, w_m):
        # Train a weak learner
        phi = AdaPerceptron(self.F, w_m)
        phi.fit(self.X, self.y, epochs, lr)
        # phi.plot_error()

        # Return weights and bias and predictions (sign)
        weights = np.concatenate((phi.W, np.array([phi.b])))
        preds = np.sign(phi.predict(self.X))

        # print(phi.error(self.y, preds))

        return weights, preds

    def compute_beta(self, w_m, phi_m_x):
        enum = w_m @ (self.y == phi_m_x)
        denom = w_m @ (self.y != phi_m_x)
        return 0.5 * np.log(enum/denom)

    def compute_w(self, w_m, phi_m_x, beta_m):
        return w_m * np.exp(- beta_m * self.y * phi_m_x)

    def tanh(self, x):
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)