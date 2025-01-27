import numpy as np
from sklearn.datasets import make_regression, make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from linear_model import ElasticNet, Lasso, Ridge, LogisticRegression
from naive_bayes import NaiveBayes
from tree import DecisionTreeClassifier
from utils import print_tree, accuracy
import matplotlib.pyplot as plt


def regressor_evalution(model):
    X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train)
    m2 = plt.scatter(X_test, y_test)
    plt.plot(X_test, y_pred, color='black', linewidth=2, label='Prediction')
    plt.show()


def classifier_evaluation(model):
    # X, y = make_classification(n_samples=200, n_features=4, n_classes=2, random_state=42)
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_score = accuracy(y_pred, y_test)
    print("Final Test Score: %.4f" % test_score)


classifier_evaluation(DecisionTreeClassifier(criterion='gini', pruning_alpha=0))
