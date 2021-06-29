import numpy as np


def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.ndarray x.
    Args:
        x: has to be an numpy.ndarray, a Matrix of dimension m * n.
    Returns:
        X as a numpy.ndarray, a vector of dimension m * (n + 1).
        None if x is not a numpy.ndarray.
        None if x is a empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) < 1 or x.shape[0] < 1:
        return None
    col = np.ones((x.shape[0],))
    return np.c_[col, x]


def vec_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.ndarray, a matrice of dimension (m, n).
        y: has to be an numpy.ndarray, a vector of dimension (m, 1).
        theta: has to be an numpy.ndarray, a vector of dimension (n, 1).
    Returns:
        The gradient as a numpy.ndarray, a vector of dimensions (n, 1), containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 2 or x.shape[0] < 1:
        return None
    if not isinstance(y, np.ndarray) or len(y.shape) != 1 or y.shape[0] != x.shape[0]:
        return None
    if not isinstance(theta, np.ndarray) or len(theta.shape) != 1 or theta.shape[0] != x.shape[1] + 1:
        return None
    m = y.shape[0]
    return (1 / m) * x.transpose().dot(add_intercept(x).dot(theta) - y)
