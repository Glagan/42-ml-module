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


def log_gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatible dimensions.
    Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x.shape) < 2 or x.shape[0] < 1 or x.shape[1] < 1:
        return None
    if len(y.shape) != 2 or y.shape[0] != x.shape[0] or y.shape[1] != 1:
        return None
    if len(theta.shape) != 2 or x.shape[1] + 1 != theta.shape[0] or theta.shape[1] != 1:
        return None
    m1 = 1 / x.shape[0]
    x_int = np.hstack((np.ones((x.shape[0], 1)), x))
    y_hat = sigmoid_(x_int.dot(theta))
    # Result vector of size n * 1
    result = []
    # Loop for each features (x0 = 1)
    for j in range(x_int.shape[1]):
        gradient_sum = 0.
        # Loop for each rows
        for index in range(x_int.shape[0]):
            y_i = y[index]
            y_hat_i = y_hat[index]
            gradient_sum = gradient_sum + (y_hat_i - y_i) * x_int[index][j]
        result.append(m1 * gradient_sum)
    return np.array(result)
