import numpy as np


def iterative_l2(theta: np.ndarray) -> float:
    """
    Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(theta, np.ndarray):
        return None
    if len(theta.shape) != 2 or theta.shape[0] < 1 or theta.shape[1] != 1:
        return None
    total = 0.
    for index in range(1, theta.shape[0]):
        total = total + theta[index].item() ** 2
    return total


def l2(theta: np.ndarray) -> float:
    """
    Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(theta, np.ndarray):
        return None
    if len(theta.shape) != 2 or theta.shape[0] < 1 or theta.shape[1] != 1:
        return None
    theta_0 = theta + 0  # copy
    theta_0[0][0] = 0
    return float(theta_0.T.dot(theta_0).item())
