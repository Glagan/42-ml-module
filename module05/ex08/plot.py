import numpy as np
import matplotlib.pyplot as plt


def plot_with_cost(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """
    Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 2 or x.shape[0] < 1 or x.shape[1] != 1:
        return None
    if not isinstance(y, np.ndarray) or y.shape != x.shape:
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (2, 1):
        return None
    xMin, xMax = min(x), max(x)
    plt.scatter(x, y, color='blue')
    plt.plot([xMin, xMax],
             [theta[0] + (theta[1] * xMin),
              theta[0] + (theta[1] * xMax)],
             color='orange')
    loss = 0.
    for index, v in enumerate(x):
        real_y = y[index][0]
        y_hat = (theta[0] + (theta[1] * v))[0]
        current_loss = real_y - y_hat
        loss = loss + (current_loss * current_loss)
        plt.plot([v, v],
                 [real_y, y_hat],
                 color='red',
                 linestyle='dashed')
    loss = (1 / y.shape[0]) * loss
    plt.title("Cost: {:.6f}".format(loss))
    plt.show()
