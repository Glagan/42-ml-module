import numpy as np
from prediction import predict_

x = np.arange(1, 6).reshape(-1, 1)

# Example 1:
theta1 = np.array([[5], [0]])
print(predict_(x, theta1))
# Ouput:
# array([5., 5., 5., 5., 5.])
# Do you remember why y_hat contains only 5's here?


# Example 2:
theta2 = np.array([[0], [1]])
print(predict_(x, theta2))
# Output:
# array([1., 2., 3., 4., 5.])
# Do you remember why y_hat == x here?


# Example 3:
theta3 = np.array([[5], [3]])
print(predict_(x, theta3))
# Output:
# array([8., 11., 14., 17., 20.])


# Example 4:
theta4 = np.array([[-3], [1]])
print(predict_(x, theta4))
# Output:
# array([-2., -1.,  0.,  1.,  2.])

# Errors

x = np.arange(1, 6).reshape(-1, 1)
theta1 = np.array([[0]])
assert predict_(x, theta1) is None, "Wrong theta should be None"
theta1 = np.array([[2.0], [3.0], [0]])
assert predict_(x, theta1) is None, "Wrong theta should be None"
assert predict_(x, 'theta') is None, "Wrong theta should be None"
theta1 = np.array([[5], [0]])
assert predict_('42', theta1) is None, "Wrong x should be None"
assert predict_([], theta1) is None, "Wrong x should be None"
assert predict_([[]], theta1) is None, "Wrong x should be None"
