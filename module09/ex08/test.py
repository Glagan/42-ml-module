import numpy as np
from my_logistic_regression import *

theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

# Example 1:
model1 = MyLogisticRegression(theta, lambda_=5.0)
print(model1.penality)
# Output
# 'l2'

print(model1.lambda_)
# Output
# 5.0

# Example 2:
model2 = MyLogisticRegression(theta, penality=None)
print(model2.penality)
# Output
# None

print(model2.lambda_)
# Output
# 0.0

# Example 3:
model3 = MyLogisticRegression(theta, penality=None, lambda_=2.0)
print(model3.penality)
# Output
# None

print(model3.lambda_)
# Output
# 0.0

print('\n---\n')

X = np.array([
    [1., 1., 2., 3.],
    [5., 8., 13., 21.],
    [3., 5., 9., 14.]
])
Y = np.array([[1], [0], [1]])
mylr = MyLogisticRegression([[2], [0.5], [7.1], [-4.3], [2.09]], penality=None)

# Example 0:
print("Model without regularization")
print("\nInitial model predictions")
print(mylr.predict_(X))
# array([[0.99930437],
#        [1.],
#        [1.]])

# Example 1:
Y1 = mylr.predict_(X)
print("loss_elem_")
print(mylr.loss_elem_(Y, Y1))

# Example 2:
print("loss_")
print(mylr.loss_(Y, Y1))
# Output:
# 11.513157421577004

# Example 3:
mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print("\nTrain model")
print(mylr.theta)
# Output:
# array([[1.04565272],
#        [0.62555148],
#        [0.38387466],
#        [0.15622435],
#        [-0.45990099]])

# Example 4:
Y2 = mylr.predict_(X)
print("Trained model predictions")
print(mylr.predict_(X))
# Output:
# array([[0.72865802],
#        [0.40550072],
#        [0.45241588]])

# Example 5:
print("loss_elem_")
print(mylr.loss_elem_(Y, Y2))

# Example 6:
print("loss_")
print(mylr.loss_(Y, Y2))
# Output:
# 0.5432466580663214

print('\n---\n')

X = np.array([
    [1., 1., 2., 3.],
    [5., 8., 13., 21.],
    [3., 5., 9., 14.]
])
Y = np.array([[1], [0], [1]])
mylr = MyLogisticRegression([[2], [0.5], [7.1], [-4.3], [2.09]])

# Example 0:
print("Model with regularization")
print("\nInitial model predictions")
print(mylr.predict_(X))

# Example 1:
Y1 = mylr.predict_(X)
print("loss_elem_")
print(mylr.loss_elem_(Y, Y1))

# Example 2:
print("loss_")
print(mylr.loss_(Y, Y1))

# Example 3:
mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print("\nTrain model")
print(mylr.theta)

# Example 4:
Y2 = mylr.predict_(X)
print("Trained model predictions")
print(mylr.predict_(X))

# Example 5:
print("loss_elem_")
print(mylr.loss_elem_(Y, Y2))

# Example 6:
print("loss_")
print(mylr.loss_(Y, Y2))
