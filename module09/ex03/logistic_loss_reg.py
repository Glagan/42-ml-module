import numpy as np


def reg_log_loss_(y: np.ndarray, y_hat: np.ndarray, theta: np.ndarray, lambda_: float) -> float:
    """
    Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for loop. The two arrays must have the same shapes.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta is empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if not isinstance(lambda_, float):
        return None
    if len(y.shape) < 2 or y.shape[0] < 1 or y.shape[1] < 1 or y.shape != y_hat.shape:
        return None
    if len(theta.shape) != 2 or theta.shape[0] < 1 or theta.shape[1] != 1:
        return None
    eps = 1e-15
    m = y.shape[0]
    one = np.ones(y.shape[0]).reshape((-1, 1))
    theta_0 = theta + 0  # copy
    theta_0[0][0] = 0
    return float((- (1 / m) * (y.T.dot(np.log(y_hat + eps)) + (one - y).T.dot(np.log(one - y_hat + eps)))) + ((lambda_ / (2 * m)) * theta_0.T.dot(theta_0)).item())
