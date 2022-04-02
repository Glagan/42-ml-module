import numpy as np


def simple_predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta shapes are not appropriate.
        None if x or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 2 or x.shape[0] < 1:
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (x.shape[1] + 1, 1):
        return None
    result = []
    for v in x:
        result.append(theta[0] + (theta[1] * v))
    return np.array(result)
