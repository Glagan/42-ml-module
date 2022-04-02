import numpy as np


def gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Computes a gradient vector from three non-empty numpy.ndarray, without any for loop. The three arrays must have compatible dimensions.
    Args:
        x: has to be a numpy.ndarray, a matrix of dimension m * 1.
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        theta: has to be a numpy.ndarray, a 2 * 1 vector.
    Returns:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta is an empty numpy.array.
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
    x = np.hstack((np.ones((x.shape[0], 1)), x))  # Add intercept
    m1 = 1 / x.shape[0]
    return (m1 * ((x.T).dot(x.dot(theta) - y)))
