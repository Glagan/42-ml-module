import numpy as np


def gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector (n + 1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of dimensions n * 1, containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x.shape) < 2 or x.shape[0] < 1 or x.shape[1] < 1:
        return None
    if len(y.shape) != 2 or y.shape[0] != x.shape[0] or y.shape[1] != 1:
        return None
    if len(theta.shape) != 2 or (x.shape[1] != theta.shape[0] and x.shape[1] + 1 != theta.shape[0]) or theta.shape[1] != 1:
        return None
    m1 = 1 / x.shape[0]
    if x.shape[1] != theta.shape[0]:
        x = np.hstack((np.ones((x.shape[0], 1)), x))
    return m1 * x.T.dot(x.dot(theta) - y)


def fit_(x: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, max_iter: int) -> np.ndarray:
    """
    Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a matrix of dimension m * n: (number of training examples, number of features).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension (n + 1) * 1: (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension (number of features + 1, 1).
        None if there is a matching shape problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x.shape) < 2 or x.shape[0] < 1 or x.shape[1] < 1:
        return None
    if len(y.shape) != 2 or y.shape[0] != x.shape[0] or y.shape[1] != 1:
        return None
    if len(theta.shape) != 2 or x.shape[1] + 1 != theta.shape[0] or theta.shape[1] != 1:
        return None
    if not isinstance(alpha, float) or not isinstance(max_iter, int):
        return None
    if alpha < 0 or max_iter < 1:
        return None
    x_int = np.hstack((np.ones((x.shape[0], 1)), x))
    for _ in range(max_iter):
        theta = theta - (alpha * gradient(x_int, y, theta))
    return theta
