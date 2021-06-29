import numpy as np


def cost_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.ndarray, without any for loop. The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.ndarray, a vector.
        y_hat: has to be an numpy.ndarray, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.ndarray.
        None if y and y_hat does not share the same dimensions.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or (len(y.shape) != 1 and y.shape[1] != 1) or y.shape[0] < 1:
        return None
    if not isinstance(y_hat, np.ndarray) or y.shape != y_hat.shape:
        return None
    m = y.shape[0]
    return (1 / (2 * m)) * ((y_hat - y).dot(y_hat - y))
