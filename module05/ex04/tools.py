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
