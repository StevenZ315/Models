import numpy as np
from linear_model import ElasticNet, Lasso, Ridge
a = np.random.rand(100, 2)
sol = np.random.rand(2,)
b = np.dot(a, sol) + 2

model = Ridge(alpha=0.1)
model.fit(a, b)
print('='*25)
print("Benchmark solution coef: ", sol.tolist())

