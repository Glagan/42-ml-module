import numpy as np


def sigmoid_(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be a numpy.ndarray, a vector.
    Returns:
        The sigmoid value as a numpy.ndarray.
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.shape == ():
        x = np.array([[x]])
    if len(x.shape) != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        return None
    return 1 / (1 + np.exp(-x))
