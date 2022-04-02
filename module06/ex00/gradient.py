import numpy as np


def simple_gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 2 or x.shape[0] < 1 or x.shape[1] != 1:
        return None
    if not isinstance(y, np.ndarray) or y.shape != x.shape:
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (2, 1):
        return None
    m1 = 1 / x.shape[0]
    sum0 = 0.0
    sum1 = 0.0
    for index, value in enumerate(x):
        y_hat = theta[0] + (theta[1] * value)
        sum0 = sum0 + (y_hat - y[index])
        sum1 = sum1 + ((y_hat - y[index]) * value)
    return np.array([m1 * sum0, m1 * sum1])
