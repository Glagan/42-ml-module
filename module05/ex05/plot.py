import numpy as np
import matplotlib.pyplot as plt


def plot(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """
    Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 2 or x.shape[0] < 1:
        return None
    if not isinstance(y, np.ndarray) or len(y.shape) < 1 or y.shape[0] != x.shape[0]:
        return None
    if not isinstance(theta, np.ndarray) or len(theta.shape) != 2 or theta.shape != (2, 1):
        return None
    xMin, xMax = min(x), max(x)
    plt.scatter(x, y, color='blue')
    plt.plot([xMin, xMax], [theta[0] + (theta[1] * xMin),
                            theta[0] + (theta[1] * xMax)], color='orange')
    plt.show()
