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


def reg_logistic_grad(x: np.ndarray, y: np.ndarray, theta: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loops.
    The three arrays must have compatible shapes.
    Args:
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if x, y or theta are empty numpy.ndarray.
        None if x, y or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if not isinstance(lambda_, int) and not isinstance(lambda_, float):
        return None
    if len(y.shape) < 2 or y.shape[0] < 1 or y.shape[1] < 1:
        return None
    if len(x.shape) < 2 or x.shape[0] != y.shape[0] or x.shape[1] < 1:
        return None
    if len(theta.shape) != 2 or theta.shape[0] != x.shape[1] + 1 or theta.shape[1] != 1:
        return None
    # Constant computations
    m1 = 1 / x.shape[0]
    theta_0 = theta + 0  # copy
    theta_0[0][0] = 0
    x_int = np.hstack((np.ones((x.shape[0], 1)), x))  # Add intercept
    y_hat = sigmoid_(x_int.dot(theta))
    # Result vector of size n * 1
    result = []
    # Loop for each features (x0 = 1)
    for j in range(x_int.shape[1]):
        gradient_sum = 0.
        # Loop for each rows
        for index in range(x_int.shape[0]):
            y_i = y[index][0]
            y_hat_i = y_hat[index][0]
            gradient_sum += (y_hat_i - y_i) * x_int[index][j]
        result.append([m1 * (gradient_sum + lambda_ * theta_0[j][0])])
    return np.array(result)


def vec_reg_logistic_grad(x: np.ndarray, y: np.ndarray, theta: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop.
    The three arrays must have compatible shapes.
    Args:
        x: has to be a numpy.ndarray, a matrix of shape m * n.
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if x, y or theta are empty numpy.ndarray.
        None if x, y or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """
    m1 = 1 / x.shape[0]
    theta_0 = theta + 0  # copy
    theta_0[0][0] = 0
    x_int = np.hstack((np.ones((x.shape[0], 1)), x))  # Add intercept
    y_hat = sigmoid_(x_int.dot(theta))
    return m1 * (x_int.T.dot(y_hat - y) + (lambda_ * theta_0))
