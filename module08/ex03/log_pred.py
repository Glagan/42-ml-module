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


def logistic_predict_(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x.shape) != 2 or len(theta.shape) != 2:
        return None
    if x.shape[0] < 1 or theta.shape[0] < 1 or x.shape[1] + 1 != theta.shape[0] or theta.shape[1] != 1:
        return None
    x_int = np.hstack((np.ones((x.shape[0], 1)), x))
    y_hat = x_int.dot(theta)
    return sigmoid_(y_hat)
