import numpy as np
import math


def mse_(y, y_hat):
    """
    Description:
        Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or (len(y.shape) != 1 and y.shape[1] != 1) or y.shape[0] < 1:
        return None
    if not isinstance(y_hat, np.ndarray) or y.shape != y_hat.shape:
        return None
    m = y.shape[0]
    return (1 / m) * ((y_hat - y).dot(y_hat - y))


def rmse_(y, y_hat):
    """
    Description:
        Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or (len(y.shape) != 1 and y.shape[1] != 1) or y.shape[0] < 1:
        return None
    if not isinstance(y_hat, np.ndarray) or y.shape != y_hat.shape:
        return None
    m = y.shape[0]
    return math.sqrt((1 / m) * ((y_hat - y).dot(y_hat - y)))


def mae_(y, y_hat):
    """
    Description:
        Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or (len(y.shape) != 1 and y.shape[1] != 1) or y.shape[0] < 1:
        return None
    if not isinstance(y_hat, np.ndarray) or y.shape != y_hat.shape:
        return None
    m = y.shape[0]
    return (1 / m) * np.sum(np.abs(y_hat - y))


def r2score_(y, y_hat):
    """
    Description:
        Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or (len(y.shape) != 1 and y.shape[1] != 1) or y.shape[0] < 1:
        return None
    if not isinstance(y_hat, np.ndarray) or y.shape != y_hat.shape:
        return None
    m = y.shape[0]
    return 1 - (np.sum((y_hat - y).dot(y_hat - y)) / np.sum((y - y.mean()).dot(y - y.mean())))
