import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initialize mean, var and priors.
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._prior = np.zeros(n_classes, dtype=np.float64)

        # Update mean var and priors
        for c in self._classes:
            X_c = X[y==c]   # Samples with y = c.
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._prior[c] = X_c.shape[0] / n_samples

    def predict(self, X):
        prediction = [self._predict(x) for x in X]
        return prediction

    def _predict(self, x):
        posteriors = []     # 后验概率
        for idx, c in enumerate(self._classes):
            prior = np.log(self._prior[c])  # 先验概率
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posteriors.append(prior + class_conditional)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x - mean)**2/(2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator / denominator

    def calc_score(self, residual):
        """
        Accuracy
        Returns:

        """
        n_samples = len(residual)
        return np.sum(residual == 0) / n_samples


