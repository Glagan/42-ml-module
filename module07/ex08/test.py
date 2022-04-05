import numpy as np
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from my_linear_regression import MyLinearRegression as MyLR

# Data
x = np.arange(1, 11).reshape(-1, 1)
y = np.array([[1.39270298],
              [3.88237651],
              [4.37726357],
              [4.63389049],
              [7.79814439],
              [6.41717461],
              [8.63429886],
              [8.19939795],
              [10.37567392],
              [10.68238222]])

# Build the model:
x_ = add_polynomial_features(x, 3)
my_lr = MyLR(np.array([[1.], [1.], [1.], [1.]]),
             alpha=0.0000005, max_iter=200000)
my_lr.fit_(x_, y)

# Plot:
# To get a smooth curve, we need a lot of data points
continuous_x = np.arange(1, 10.01, 0.01).reshape(-1, 1)
x_ = add_polynomial_features(continuous_x, 3)

plt.scatter(x, y)
plt.plot(continuous_x, my_lr.predict_(x_), color='orange')
plt.show()
