# Author: Steven Zhao

import numpy as np

class LinearRegression:
    """
    Linear Regression model with L1/L2 regulation.

    Args:
        learning_rate: float, optional, default 0.001
            Learning rate for gradient descent
        n_iters: int, optional, default 1000
            Number of iterations
        penalty: None | 'L1' | 'L2'
            Whether to apply L1 or L2 regulation
        c: float, optional, default 0.05
            penalty term hyper-parameter.

    Attributes:
        weight_: array, (n_samples,)
            parameter vector (w in the cost function)
        bias_: float
            independent term in decision function
        history_: array, (n_iters, )
            MSE results during iteration.

    Example:
    ----------------------------------
        >> import base_model
        >> clf = base_model.LinearRegression(penalty='L2')
        >> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
        >> y = clf.predict([[0,0], [1, 1], [2, 2]])

    """
    def __init__(self, learning_rate=0.01, n_iters=1000, penalty=None, c=0.05):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.penalty = penalty
        self.c = c
        self.weight_ = None
        self.bias_ = None
        self.history_ = []

    def fit(self, X, y):
        """Fit model with X, y

        Calculate weight and bias from X, y with gradient descent.

        Args:
            X: ndarray, (n_samples, n_features)
                Input data
            y: ndarray, (n_samples,)
                Target result.

        Returns:
            None
        """
        n_samples, n_features = X.shape

        # Initialize weight and bias.
        self.weight_ = np.random.rand(n_features)
        self.bias_ = 0

        # Gradient descent
        for iter in range(self.n_iters):
            y_pred = np.dot(X, self.weight_) + self.bias_
            residual = y_pred - y
            self.history_.append(self.mse(residual))

            # Print residual info every 100 iterations.
            if (iter + 1) % 100 == 0:
                print('Loss for iteration %d is %.4f.' % (iter + 1, self.history_[-1]))

            # Calculate derivative of weight and bias.
            d_weight = (1/n_samples) * np.dot(X.T, residual)
            d_bias = (1/n_samples) * np.sum(residual)

            # L1 Regulation
            #
            if self.penalty == 'L1':
                weight_sign = np.array([1 if weight > 0 else -1 for weight in self.weight_])
                bias_sign = 1 if self.bias_ > 0 else -1
                d_weight += self.c * weight_sign
                d_bias += self.c * bias_sign


            # L2 Regulation
            if self.penalty == 'L2':
                d_weight += self.c * self.weight_
                d_bias += self.c * self.bias_

            # Update weight and bias
            self.weight_ -= self.learning_rate * d_weight
            self.bias_ -= self.learning_rate * d_bias

    def predict(self, X):
        """Predict y using the linear model

        Args:
            X: ndarray, (n_samples, n_features)
                Input data

        Returns:
            y: array, (n_samples,)
                Predicted values
        """
        return np.dot(X, self.weight_) + self.bias_

    def mse(self, residual):
        """Calculate Mean squared error between two arrays.

        Args:
            residual: array, (n_samples,)
                array of y_pred - y
        Returns:
            mse: float
        """
        n_samples = len(residual)
        return (1/n_samples) * np.sum(residual**2)
