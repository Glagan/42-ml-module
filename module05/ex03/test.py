import numpy as np
from tools import add_intercept

# Example 1:
x = np.arange(1, 6).reshape((5, 1))
print(add_intercept(x))
# Output:
# array([[1., 1.],
#        [1., 2.],
#        [1., 3.],
#        [1., 4.],
#        [1., 5.]])


# Example 2:
y = np.arange(1, 10).reshape((3, 3))
print(add_intercept(y))
# Output:
# array([[1., 1., 2., 3.],
#        [1., 4., 5., 6.],
#        [1., 7., 8., 9.]])

# Example 3:
z = np.ndarray((4, 2))
print(add_intercept(z))

# Errors
assert add_intercept('42') is None, "Invalid matrix should be None"
assert add_intercept([]) is None, "Invalid matrix should be None"
assert add_intercept(np.ndarray(
    (74,))) is None, "Invalid matrix should be None"
