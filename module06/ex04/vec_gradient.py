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
    if not isinstance(x, np.ndarray) or len(y.shape) != 1:
        return None
    if not isinstance(y, np.ndarray) or y.shape != x.shape:
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (2,):
        return None
    col = np.ones((x.shape[0],))
    x_int = np.c_[col, x]
    m1 = 1 / x.shape[0]
    return theta - (m1 * ((x_int.transpose()).dot(x_int.dot(theta) - y)))
