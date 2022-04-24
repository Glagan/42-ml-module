import numpy as np


def add_polynomial_features(x: np.ndarray, power: int) -> np.ndarray:
    """
    Add polynomial features to vector x by raising its values up to the power given in argument.  
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        power: has to be an int, the power up to which the components of vector x are going to be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray, of dimension m * (n * p), containg he polynomial feature values for all training examples.
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if len(x.shape) != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        return None
    if not isinstance(power, int) or power < 1:
        return None
    result = x.copy()
    for i in range(2, power + 1):
        result = np.hstack((result, np.power(x, i).reshape(-1, x.shape[1])))
    return result
