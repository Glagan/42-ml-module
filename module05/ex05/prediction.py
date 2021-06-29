import numpy as np


def predict_(x: np.ndarray, theta: np.ndarray):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 1 or x.shape[0] < 1:
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (2,):
        return None
    col = np.ones((x.shape[0],))
    return np.dot(np.c_[col, x], theta)
