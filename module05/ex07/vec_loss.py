import numpy as np


def loss_(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        None if y or y_hat is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray):
        return None
    if not isinstance(y_hat, np.ndarray):
        return None
    if len(y.shape) != 2 or y.shape[0] < 1 or y.shape != y_hat.shape:
        return None
    m2 = 1 / (2 * y.shape[0])
    return float((m2 * (y_hat - y).T.dot(y_hat - y)).item())
