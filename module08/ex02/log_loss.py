import math
import numpy as np


def log_loss_(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-15) -> float:
    """
    Computes the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if len(y.shape) != 2 or y.shape[0] < 1 or y.shape[1] != 1:
        return None
    if y.shape != y_hat.shape:
        return None
    if not isinstance(eps, float) or eps == 0:
        return None
    m1 = 1 / y.shape[0]
    loss_sum = 0.
    for i in range(y.shape[0]):
        y_i = y[i][0]
        y_hat_i = y_hat[i][0]
        loss_sum = loss_sum + (y_i * math.log(y_hat_i + eps)) + ((1 - y_i) * math.log(1 - y_hat_i + eps))
    return - m1 * loss_sum
