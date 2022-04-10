import numpy as np


def vec_log_loss_(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-15) -> float:
    """
    Compute the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: epsilon (default=1e-15)
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
    one = np.ones(y.shape[0]).reshape((-1, 1))
    m1 = 1 / y.shape[0]
    return float((- m1 * (y.T.dot(np.log(y_hat + eps)) + (one - y).T.dot(np.log(one - y_hat + eps)))).item())
