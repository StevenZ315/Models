# Author: Steven Zhao

import numpy as np
from collections import defaultdict


class LinearModel:
    """Base class for Linear Models"""
    def fit(self, X, y):
        pass

    def _decision_function(self, X):
        return np.dot(X, self.coef_.T) + self.intercept_

    def predict(self, X):
        return self._decision_function(X)

    def __str__(self):
        pass


class ElasticNet(LinearModel):
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, learning_rate=0.01, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.l2_ratio = 1 - l1_ratio
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol

        self.coef_ = None
        self.intercept_ = None
        self.score_ = None

        self.history_ = defaultdict(list)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize coef and intercept.
        self.coef_ = np.random.rand(n_features)
        self.intercept_ = 0

        iteration = 0
        while iteration < self.max_iter:
            iteration += 1
            self.gradient_descent(X, y)

            if (iteration + 1) % 10 == 0:
                print('Iteration: %d\tTrain Score: %.4f' % (iteration, self.score_))

            # Early stopping.
            if self.score_ < self.tol:
                print("Designed precision reached at iteration %d." % iteration)

        print("Final score: %.4f" % self.score_)


    def gradient_descent(self, X, y):
        n_samples, n_features = X.shape

        y_pred = self.predict(X)
        residual = y_pred - y
        self.score_ = self.calc_score(residual)

        # Calculate derivatives of weights and bias.
        d_coef = (1/n_samples) * np.dot(X.T, residual)
        d_intercept = (1/n_samples) * np.sum(residual)

        # L1 regulation.
        coef_sign = np.sign(self.coef_)
        l1_reg_coef = self.learning_rate * coef_sign
        bias_sign = 1 if self.intercept_ > 0 else -1
        l1_reg_intercept = self.learning_rate * self.intercept_ * bias_sign

        # L2 regulation.
        l2_reg_coef = self.learning_rate * self.coef_
        l2_reg_intercept = self.learning_rate * self.intercept_

        # Update weight and bias.
        d_coef += (self.l1_ratio * l1_reg_coef + self.l2_ratio * l2_reg_coef)
        d_intercept += (self.l1_ratio * l1_reg_intercept + self.l2_ratio * l2_reg_intercept)

        self.coef_ -= self.alpha * d_coef
        self.intercept_ -= self.alpha * d_intercept

        # Save the current calculation.
        self.update_history()


    def update_history(self):
        self.history_['coef'].append(self.coef_.tolist())
        self.history_['intercept'].append(self.intercept_.tolist())
        self.history_['score'].append(self.score_)

    def calc_score(self, residual):
        """
        Mean squared error.
        Returns:

        """
        n_samples = len(residual)
        return (1/n_samples) * np.sum(residual**2)


class Lasso(ElasticNet):
    def __init__(self, **kwargs):
        super().__init__(l1_ratio=1, **kwargs)


class Ridge(ElasticNet):
    def __init__(self, **kwargs):
        super().__init__(l1_ratio=0, **kwargs)


class LogisticRegression(ElasticNet):
    def __init__(self, **kwargs):
        super().__init__(learning_rate=0, **kwargs)

    def _decision_function(self, X):
        linear_model = np.dot(X, self.coef_.T) + self.intercept_
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted > 0.5).astype(int)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calc_score(self, residual):
        """
        Accuracy
        Returns:

        """
        n_samples = len(residual)
        return np.sum(residual == 0) / n_samples
