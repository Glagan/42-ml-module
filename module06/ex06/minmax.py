import numpy as np


def minmax(x: np.ndarray) -> np.ndarray:
    """
    Computes the normalized version of a non-empty numpy.array using the min-max standardization.
    Args:
        x: has to be an numpy.array, a vector.
    Return:
        x' as a numpy.array.
        None if x is a non-empty numpy.array or not a numpy.array.
        None if x is not of the expected type.
    Raises:
        This function shouldn't raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if len(x.shape) < 1 or any([size < 1 for size in x.shape]):
        return None
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)
