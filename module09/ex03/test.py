import numpy as np
from logistic_loss_reg import reg_log_loss_


y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
# array([[1],
#        [1],
#        [0],
#        [0],
#        [1],
#        [1],
#        [0]])
y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
# array([[0.9 ],
#        [0.79],
#        [0.12],
#        [0.04],
#        [0.89],
#        [0.93],
#        [0.01]])
theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
# array([[ 1. ],
#        [ 2.5],
#        [ 1.5],
#        [-0.9]])

# Example :
print(reg_log_loss_(y, y_hat, theta, .5))
# Output:
# 0.43377043716475955

# Example :
print(reg_log_loss_(y, y_hat, theta, .05))
# Output:
# 0.13452043716475953

# Example :
print(reg_log_loss_(y, y_hat, theta, .9))
# Output:
# 0.6997704371647596
