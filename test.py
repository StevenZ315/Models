import base_model
import numpy as np

input = [[0, 0], [1, 1], [2, 2]]
output = [0, 1, 2]

X = np.array(input)
y = np.array(output)

clf = base_model.LinearRegression(penalty='L1', c=0)
clf.fit(X, y)
print(clf.weight_)
print(clf.bias_)