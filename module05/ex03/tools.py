import numpy as np


def add_intercept(x: np.ndarray) -> np.ndarray:
    """
    Adds a column of 1's to the non-empty numpy.array x.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
    Returns:
        x as a numpy.array, a vector of shape m * 2.
        None if x is not a numpy.array.
        None if x is a empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 2 or x.shape[0] < 1:
        return None
    return np.hstack((np.ones((x.shape[0], 1)), x))
