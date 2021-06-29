import numpy as np
import matplotlib.pyplot as plt


def plot_with_cost(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 1 or x.shape[0] < 1:
        return None
    if not isinstance(y, np.ndarray) or y.shape[0] != x.shape[0]:
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (2,):
        return None
    xMin, xMax = min(x), max(x)
    plt.scatter(x, y, color='blue')
    plt.plot([xMin, xMax], [theta[0] + (theta[1] * xMin),
                            theta[0] + (theta[1] * xMax)], color='orange')
    for index, v in enumerate(x):
        plt.plot([v, v],
                 [y[index], theta[0] + (theta[1] * v)],
                 color='red', linestyle='dashed')
    plt.show()
