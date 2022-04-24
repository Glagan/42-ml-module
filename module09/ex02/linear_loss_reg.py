import numpy as np


def reg_loss_(y: np.ndarray, y_hat: np.ndarray, theta: np.ndarray, lambda_: float) -> float:
    """
    Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop. The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta are empty numpy.ndarray.
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
    theta_0 = theta + 0  # copy
    theta_0[0][0] = 0
    diff = (y_hat - y).T.dot(y_hat - y)  # (y_hat − y) . (y_hat − y)
    return float(((1 / (2 * y.shape[0])) * (diff + lambda_ * theta_0.T.dot(theta_0))).item())
