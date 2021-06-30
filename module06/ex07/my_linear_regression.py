import numpy as np


class MyLinearRegression:
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """

    def __init__(self, theta, alpha=0.00001, max_iter=100000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    def gradient_(self, x: np.ndarray, y: np.ndarray):
        """Computes a gradient vector from three non-empty numpy.ndarray, without any for loop. The three arrays must have compatible dimensions.
        Args:
            x: has to be a numpy.ndarray, a matrix of dimension m * 1.
            y: has to be a numpy.ndarray, a vector of dimension m * 1.
            theta: has to be a numpy.ndarray, a 2 * 1 vector.
        Returns:
            The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
            None if x, y, or theta is an empty numpy.ndarray.
            None if x, y and theta do not have compatible dimensions.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        m1 = 1 / x.shape[0]
        return m1 * x.transpose().dot(x.dot(self.theta) - y)

    def fit_(self, x: np.ndarray, y: np.ndarray):
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during the gradient descent
        Returns:
            new_theta: numpy.ndarray, a vector of dimension 2 * 1.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        if y.shape != x.shape:
            return None
        col = np.ones((x.shape[0],))
        x_int = np.c_[col, x]
        for it in range(self.max_iter):
            self.theta = self.theta - \
                (self.alpha * self.gradient_(x_int, y))
        return self.theta

    def predict_(self, x: np.ndarray):
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
        if not isinstance(x, np.ndarray):
            return None
        col = np.ones((x.shape[0],))
        return np.dot(np.c_[col, x], self.theta)

    def cost_elem_(self, x: np.ndarray, y: np.ndarray):
        """
        Description:
            Calculates all the elements (1 / (2 * M)) * (y - x)^2 of the cost function.
        Args:
            x: has to be an numpy.ndarray, a vector.
            y: has to be an numpy.ndarray, a vector.
        Returns:
            J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
            None if there is a dimension matching problem between X, Y or theta.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        m = x.shape[0]
        return (1 / (2 * m)) * ((y - x) ** 2)

    def cost_(self, x: np.ndarray, y: np.ndarray):
        """
        Description:
            Calculates the value of cost function.
        Args:
            x: has to be an numpy.ndarray, a vector.
            y: has to be an numpy.ndarray, a vector.
        Returns:
            J_value : has to be a float.
            None if there is a dimension matching problem between X, Y or theta.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        m = y.shape[0]
        return (1 / (2 * m)) * np.sum((y - x) ** 2)
