import numpy as np


def add_intercept(x: np.ndarray) -> np.ndarray:
    """
    Adds a column of 1's to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        return None
    return np.hstack((np.ones((x.shape[0], 1)), x))
