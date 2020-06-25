import numpy as np


def accuracy(y_pred, y_actual):
    n_samples = len(y_pred)
    return np.sum(y_pred == y_actual) / n_samples
