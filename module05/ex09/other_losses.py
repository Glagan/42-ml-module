import numpy as np
import math


def mse_(y: np.ndarray, y_hat: np.ndarray) -> float:
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
    if not isinstance(y, np.ndarray):
        return None
    if not isinstance(y_hat, np.ndarray):
        return None
    if len(y.shape) != 2 or y.shape[0] < 1 or y.shape != y_hat.shape:
        return None
    m = y.shape[0]
    return (1 / m) * np.sum(np.square(y_hat - y))


def rmse_(y: np.ndarray, y_hat: np.ndarray) -> float:
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
    if not isinstance(y, np.ndarray):
        return None
    if not isinstance(y_hat, np.ndarray):
        return None
    if len(y.shape) != 2 or y.shape[0] < 1 or y.shape != y_hat.shape:
        return None
    m = y.shape[0]
    return math.sqrt((1 / m) * np.sum(np.square(y_hat - y)))


def mae_(y: np.ndarray, y_hat: np.ndarray) -> float:
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
    if not isinstance(y, np.ndarray):
        return None
    if not isinstance(y_hat, np.ndarray):
        return None
    if len(y.shape) != 2 or y.shape[0] < 1 or y.shape != y_hat.shape:
        return None
    m = y.shape[0]
    return (1 / m) * np.sum(np.abs(y_hat - y))


def r2score_(y: np.ndarray, y_hat: np.ndarray) -> float:
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
    if not isinstance(y, np.ndarray):
        return None
    if not isinstance(y_hat, np.ndarray):
        return None
    if len(y.shape) != 2 or y.shape[0] < 1 or y.shape != y_hat.shape:
        return None
    return 1 - (np.sum(np.square(y_hat - y)) / np.sum(np.square(y - y.mean())))
