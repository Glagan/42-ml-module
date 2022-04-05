import numpy as np
from my_linear_regression import MyLinearRegression as MyLR

X = np.array([[1., 1., 2., 3.],
              [5., 8., 13., 21.],
              [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLR(np.array([[1.], [1.], [1.], [1.], [1.]]))

# Example 0:
print(mylr.predict_(X))
# Output:
# array([[8.], [48.], [323.]])

# Example 1:
Y1 = mylr.predict_(X)
print(mylr.loss_elem_(Y, Y1))
# Output:
# array([[225.], [0.], [11025.]])

# Example 2:
print(mylr.loss_(Y, Y1))
# Output:
# 1875.0

# Example 3:
mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print(mylr.theta)
# Output:
# array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])
# or
# [[ 1.81883792e+01]
#  [ 2.76697788e+00]
#  [-3.74782024e-01]
#  [ 1.39219585e+00]
#  [ 1.74138279e-02]]

# Example 4:
Y2 = mylr.predict_(X)
print(mylr.predict_(X))
# Output:
# array([[23.417..], [47.489..], [218.065...]])

# Example 5:
print(mylr.loss_elem_(Y, Y2))
# Output:
# array([[0.174..], [0.260..], [0.004..]])

# Example 6:
print(mylr.loss_(Y, Y2))
# Output:
# 0.0732..
