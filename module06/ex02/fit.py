import numpy as np


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, without any for loop. The three arrays must have compatible dimensions.
    Args:
        x: has to be a numpy.ndarray, a matrix of dimension m * 1.
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        theta: has to be a numpy.ndarray, a 2 * 1 vector.
    Returns:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    m1 = 1 / x.shape[0]
    return m1 * x.transpose().dot(x.dot(theta) - y)


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    if not alpha or alpha < 0:
        return None
    if max_iter < 1:
        return None
    col = np.ones((x.shape[0],))
    x_int = np.c_[col, x]
    new_theta = theta
    for it in range(max_iter):
        new_theta = (new_theta - (alpha * gradient(x_int, y, new_theta)))
    return new_theta
