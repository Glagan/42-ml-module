import numpy as np


def loss_elem_(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """
    Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between y and y_hat.
        None if y or y_hat is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray):
        return None
    if not isinstance(y_hat, np.ndarray):
        return None
    if len(y.shape) != 2 or y.shape[0] < 1 or y.shape != y_hat.shape:
        return None
    return np.square(y_hat - y)


def loss_(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Description:
        Calculates the value of loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_value : has to be a float.
        None if there is a shape matching problem between y or y_hat.
        None if y or y_hat is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray):
        return None
    if not isinstance(y_hat, np.ndarray):
        return None
    if len(y.shape) != 2 or y.shape[0] < 1 or y.shape != y_hat.shape:
        return None
    return (1 / (2 * y.shape[0])) * np.sum(loss_elem_(y, y_hat))
