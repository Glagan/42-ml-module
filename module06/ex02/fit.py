import numpy as np


def gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Computes a gradient vector from three non-empty numpy.ndarray, without any for loop. The three arrays must have compatible dimensions.
    Args:
        x: has to be a numpy.ndarray, a matrix of dimension m * 2.
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        theta: has to be a numpy.ndarray, a 2 * 1 vector.
    Returns:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta is an empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 2 or x.shape[0] < 1 or x.shape[1] != 2:
        return None
    if not isinstance(y, np.ndarray) or len(y.shape) != 2 or y.shape[0] != x.shape[0] or y.shape[1] != 1:
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (2, 1):
        return None
    m1 = 1 / x.shape[0]
    return (m1 * ((x.T).dot(x.dot(theta) - y)))


def fit_(x: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, max_iter: int):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
        y: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
        theta: has to be a numpy.array, a vector of shape 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of shape 2 * 1.
        None if there is a matching shape problem.
        None if x, y, theta, alpha or max_iter is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 2 or x.shape[0] < 1 or x.shape[1] != 1:
        return None
    if not isinstance(y, np.ndarray) or y.shape != x.shape:
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (2, 1):
        return None
    if not isinstance(alpha, float) or not alpha or alpha < 0:
        return None
    if not isinstance(max_iter, int) or max_iter < 1:
        return None
    if max_iter < 1:
        return None
    x = np.hstack((np.ones((x.shape[0], 1)), x))  # Add intercept
    new_theta = theta
    for _ in range(max_iter):
        new_theta = (new_theta - (alpha * gradient(x, y, new_theta)))
    return new_theta
