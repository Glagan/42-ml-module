import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from other_losses import *

# Example 1:
x = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

# Mean squared error
# your implementation
print(mse_(x, y))
# Output:
# 4.285714285714286
# > sklearn implementation
print(mean_squared_error(x, y))
# Output:
# 4.285714285714286

# Root mean squared error
# your implementation
print(rmse_(x, y))
# Output:
# 2.0701966780270626
# > sklearn implementation not available: take the square root of MSE
print(sqrt(mean_squared_error(x, y)))
# Output:
# 2.0701966780270626

# Mean absolute error
# your implementation
print(mae_(x, y))
# Output:
# 1.7142857142857142
# > sklearn implementation
print(mean_absolute_error(x, y))
# Output:
# 1.7142857142857142

# R2-score
# your implementation
print(r2score_(x, y))
# Output:
# 0.9681721733858745
# sklearn implementation
print(r2_score(x, y))
# Output:
# 0.9681721733858745

# Errors
assert mse_('42', y) is None, "Invalid call should be None"
assert rmse_('42', y) is None, "Invalid call should be None"
assert mae_('42', y) is None, "Invalid call should be None"
assert r2score_('42', y) is None, "Invalid call should be None"
assert mse_(x, '42') is None, "Invalid call should be None"
assert rmse_(x, '42') is None, "Invalid call should be None"
assert mae_(x, '42') is None, "Invalid call should be None"
assert r2score_(x, '42') is None, "Invalid call should be None"
z = np.array([2, 14, -13, 5, 12, 4, -19])
assert mse_(z, y) is None, "Invalid call should be None"
assert rmse_(z, y) is None, "Invalid call should be None"
assert mae_(z, y) is None, "Invalid call should be None"
assert r2score_(z, y) is None, "Invalid call should be None"
assert mse_(x, z) is None, "Invalid call should be None"
assert rmse_(x, z) is None, "Invalid call should be None"
assert mae_(x, z) is None, "Invalid call should be None"
assert r2score_(x, z) is None, "Invalid call should be None"
