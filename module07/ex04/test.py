import numpy as np
from fit import fit_


def predict_(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Computes the prediction vector y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x.shape) != 2 or len(theta.shape) != 2 or x.shape[1] + 1 != theta.shape[0] or theta.shape[1] != 1:
        return None
    x_int = np.hstack((np.ones((x.shape[0], 1)), x))
    return x_int.dot(theta)


x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta = np.array([[42.], [1.], [1.], [1.]])

# Example 0:
theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
print(theta2)
# Output:
# array([[41.99..],[0.97..], [0.77..], [-1.20..]])

# Example 1:
print(predict_(x, theta2))
# Output:
# array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])
