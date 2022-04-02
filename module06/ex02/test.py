import numpy as np
from fit import fit_

x = np.array([[12.4956442], [21.5007972],
              [31.5527382], [48.9145838],
              [57.5088733]])
y = np.array([[37.4013816], [36.1473236],
              [45.7655287], [46.6793434],
              [59.5585554]])
theta = np.array([[1], [1]])

# Example 0:
theta1 = fit_(x, y, theta, alpha=5e-6, max_iter=15000)
print(theta1)
# Output:
# array([[1.40709365], [1.1150909 ]])


def predict_(x: np.ndarray, theta: np.ndarray):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """
    if not isinstance(x, np.ndarray) or len(x.shape) != 2 or x.shape[0] < 1:
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (2, 1):
        return None
    with_intercept = np.hstack((np.ones((x.shape[0], 1)), x))
    return np.dot(with_intercept, theta)


# Example 1:
print(predict_(x, theta1))
# Output:
# array([[15.3408728],
#        [25.38243697],
#        [36.59126492],
#        [55.95130097],
#        [65.53471499]])
