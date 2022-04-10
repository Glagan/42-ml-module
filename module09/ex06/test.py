import numpy as np
from ridge import MyRidge

X = np.array([[1., 1., 2., 3.],
              [5., 8., 13., 21.],
              [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
model = MyRidge(np.array([[1.], [1.], [1.], [1.], [1.]]))

# Params
# Get
print(model.get_params_())
# Output:
# {
#     "thetas": array([[1.], [1.], [1.], [1.], [1.]]),
#     "alpha": 0.001,
#     "max_iter": 42000,
#     "lambda_": 0.5,
# }

# Set
model.set_params_({
    "lambda_": 0.6
})
print(model.get_params_())
# Output:
# {
#     "thetas": array([[1.], [1.], [1.], [1.], [1.]]),
#     "alpha": 0.001,
#     "max_iter": 42000,
#     "lambda_": 0.6,
# }

# Example 0:
print("\nPrediction with initial parameters")
Y1 = model.predict_(X)
print(Y1)

# Example 1:
print("loss_elem_")
print(model.loss_elem_(Y, Y1))

# Example 2:
print("loss_")
print(model.loss_(Y, Y1))

# Example 3:
print("\nTrain model")
model.alpha = 1.6e-4
model.max_iter = 200000
model.fit_(X, Y)
print("thetas")
print(model.thetas)

# Example 4:
print("\nPredictions with trained model")
Y2 = model.predict_(X)
print(model.predict_(X))

# Example 5:
print("loss_elem_")
print(model.loss_elem_(Y, Y2))

# Example 6:
print("loss_")
print(model.loss_(Y, Y2))
