import numpy as np


def simple_predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Computes the prediction vector y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x.shape) != 2 or len(theta.shape) != 2 or (x.shape[1] + 1) != theta.shape[0] or theta.shape[1] != 1:
        return None
    result = []
    for row in x:
        row_result = 0
        for i, v in enumerate(row):
            row_result += (v * theta[i + 1])
        result.append(row_result + theta[0])
    return np.array(result)
