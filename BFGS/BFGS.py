import numpy as np

class BFGSDenoiser():
    def __init__(self, nrows, ncols):
        """
        Randomly initialized the denoising filter.

        args:
            nrows: Height of the filter.
            ncols: Width of the filter.
        """
        self.f = np.random.rand(nrows * ncols) * 0.1

        self.nrows = nrows
        self.ncols = ncols
        self.error_prog = []

    def get_x(self, img):
        """
        Takes patches of the input image via a sliding window, flattens and stacks them into a matrix.
        Multiplying the (flattened) filter with the output is equivalent to convolution.
        """
        img = self.pad_input(img)

        patches = []

        for i in range(img.shape[0] - self.nrows + 1):
            for j in range(img.shape[1] - self.ncols + 1):
                patch = img[i:i+self.nrows, j:j+self.ncols]
                patches.append(patch.flatten())
        X = np.stack(patches, axis = 1)
        return X

    def pad_input(self, inp):
        """
        Pads the input image on all sides to maintain the original size in the output ("same" padding).
        """
        p_rows = (self.nrows - 1) // 2
        p_cols = (self.ncols - 1) // 2

        # Add an offset on one side if there is an even number of rows/columns
        h_offset = 1 if self.nrows % 2 == 0 else 0
        w_offset = 1 if self.ncols % 2 == 0 else 0

        return np.pad(inp, pad_width=((p_rows, p_rows + h_offset), (p_cols, p_cols + w_offset)))

    def apply_filter(self, X):
        """
        Applies the learned filter to X and a sigmoid activation to keep pixel values between 0 and 1.
        """
        return self.sigmoid(self.f @ X)

    def get_error(self, y_pred, y):
        """
        Computes the mean squared error.
        """
        N = y.shape[0]
        y = self.sigmoid(y)

        return (1/N) * ((y - y_pred) @ (y - y_pred))

    def get_gradient(self, f, X, y):
        """
        Computes the gradient of the error for weight updates.
        """
        N = y.shape[0]
        y = self.sigmoid(y)
        y_pred = self.sigmoid(f @ X)

        return - (2/N) * X @ ((y - y_pred) * (y_pred * (1 - y_pred)))

    def sigmoid(self, x):
        return 1 / (1 + np.exp((-1)*x))

    def fit(self, X, y, epochs = 10, lr = 0.1, verbose = False):
        # Get the patches
        X = self.get_x(X)
        y = y.flatten()

        # Initialize G as an identity matrix
        self.G = np.eye(len(self.f))

        for epoch in range(epochs):
            # Apply the filter to the input
            y_pred = self.apply_filter(X)

            # Calculate the error
            error = self.get_error(y_pred, y)
            self.error_prog.append(error)

            f_grad = self.get_gradient(self.f, X, y)

            # Calculate the search direction
            direction = -self.G @ f_grad

            # Calculate the new f and its derivative
            f_1 = self.f + lr * direction
            f_1_grad = self.get_gradient(f_1, X, y)

            p = f_1 - self.f
            v = f_1_grad - f_grad

            u = p / (p.T @ v) - (self.G @ v) / (v.T @ self.G @ v)

            pTv = p.T @ v
            if pTv > 1e-10: # To avoid division by zero
                Gv = self.G @ v
                self.G = self.G + np.outer(p, p) / pTv - np.outer(Gv, Gv) / (v.T @ Gv) + (v.T @ Gv) * np.outer(u, u)

            # Update filter
            self.f = f_1

            if verbose:
                print(f"Epoch {epoch + 1} | Error (* 1e6): {error * 100000:.2f}")

    def predict(self, X):
        # Save shape of the original image
        height = X.shape[0]
        width = X.shape[1]

        # Get the patches
        X = self.get_x(X)

        # Apply the filter
        y_pred = self.apply_filter(X)

        # Reshape back into two dimensions
        y_pred = y_pred.reshape(height, width)

        return y_pred

    def get_filter(self):
        """
        Reshapes the flattened filter into a 2D array for the purposes of visualization.
        """
        return self.f.reshape(self.nrows, self.ncols)