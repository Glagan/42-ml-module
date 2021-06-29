import numpy as np


def cost_elem_(y, y_hat):
    """
    Description:
        Calculates all the elements (1/2*M)*(y_pred - y)^2 of the cost function.
    Args:
        y: has to be an numpy.ndarray, a vector.
        y_hat: has to be an numpy.ndarray, a vector.
    Returns:
        J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
    Raises:
        This function should not raise any Exception.
    """
    m = y.shape[0]
    return (1 / (2 * m)) * ((y_hat - y) * (y_hat - y))


def cost_(y, y_hat):
    """
    Description:
        Calculates the value of cost function.
    Args:
        y: has to be an numpy.ndarray, a vector.
        y_hat: has to be an numpy.ndarray, a vector.
    Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or (len(y.shape) != 1 and y.shape[1] != 1) or y.shape[0] < 1:
        return None
    if not isinstance(y_hat, np.ndarray) or (len(y_hat.shape) != 1 and y_hat.shape[1] != 1) or y_hat.shape[0] < 1:
        return None
    result = []
    for index, v in enumerate(y):
        result.append((y_hat[index] - v).squeeze() ** 2)
    m = y.shape[0]
    return (1 / (2 * m)) * sum(result)
