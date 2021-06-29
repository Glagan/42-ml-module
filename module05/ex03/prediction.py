import numpy as np


def simple_predict(x: np.ndarray, theta: np.ndarray):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 1 or x.shape[0] < 1:
        return None
    if not isinstance(x, np.ndarray) or theta.shape != (2,):
        return None
    result = []
    for v in x:
        result.append(theta[0] + (theta[1] * v))
    return np.array(result)
