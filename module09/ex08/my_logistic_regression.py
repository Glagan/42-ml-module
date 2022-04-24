
import numpy as np


class MyLogisticRegression:
    """
    My personnal logistic regression to classify things.
    It works great !
    """

    supported_penalities = ['l2']

    def __init__(self, theta: np.ndarray, alpha: float = 0.0005, max_iter: int = 42000, penality: str = 'l2', lambda_: float = 0.5) -> np.ndarray:
        if not isinstance(theta, np.ndarray):
            try:
                theta = np.array(theta)
            except BaseException:
                raise TypeError("theta must be a compatible np.ndarray")
        if len(theta.shape) != 2 or theta.shape[0] < 2 or theta.shape[1] != 1:
            raise ValueError("theta must be a vector of at size (n, 1) with n >= 2")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be a positive number")
        if not isinstance(max_iter, int):
            raise TypeError("max_iter must be an int")
        if max_iter < 1:
            raise ValueError("max_iter must be a positive number")
        if not isinstance(penality, str) and penality is not None:
            raise TypeError(f"penality should be a str or None")
        if penality is not None and penality not in self.supported_penalities:
            raise ValueError(f"penality should be in [{', '.join(self.supported_penalities)}] or None")
        if not isinstance(lambda_, float) and not isinstance(lambda_, int):
            raise TypeError('lambda_ should be a positive number')
        if lambda_ < 0:
            raise ValueError('lambda_ should be a positive number')
        self.lambda_ = lambda_
        self.theta = theta
        self.alpha = alpha
        self.max_iter = max_iter
        self.penality = penality
        self.lambda_ = lambda_ if penality in self.supported_penalities else 0.

    def sigmoid_(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid of a vector.
        Args:
            x: has to be a numpy.ndarray, a vector.
        Returns:
            The sigmoid value as a numpy.ndarray.
            None if x is an empty numpy.ndarray.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray):
            return None
        if x.shape == ():
            x = np.array([[x]])
        if len(x.shape) != 2 or x.shape[0] < 1 or x.shape[1] < 1:
            return None
        return 1 / (1 + np.exp(-x))

    def predict_(self, x: np.ndarray):
        if not isinstance(x, np.ndarray):
            return None
        if len(x.shape) != 2:
            return None
        if x.shape[0] < 1 or x.shape[1] + 1 != self.theta.shape[0]:
            return None
        x_int = np.hstack((np.ones((x.shape[0], 1)), x))
        y_hat = x_int.dot(self.theta)
        return self.sigmoid_(y_hat)

    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """ 
        Calculates all the elements loss.
        Args:
            y: has to be an numpy.ndarray, a vector.
            y_hat: has to be an numpy.ndarray, a vector.
        Returns:
            J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if len(y.shape) < 2 or y.shape[0] < 1 or y.shape[1] != 1:
            return None
        if y.shape != y_hat.shape:
            return None
        eps = 1e-15
        one = np.ones(y.shape[0]).reshape((-1, 1))
        if self.penality == 'l2':
            m = y.shape[0]
            theta_0 = self.theta + 0  # copy
            theta_0[0][0] = 0
            return (y * np.log(y_hat + eps)) + ((one - y) * np.log(one - y_hat + eps)) + (self.lambda_ / (2 * m)) * theta_0.T.dot(theta_0)
        return (y * np.log(y_hat + eps)) + ((one - y) * np.log(one - y_hat + eps))

    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Compute the logistic loss value.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        Returns:
            The logistic loss value as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if len(y.shape) < 2 or y.shape[0] < 1 or y.shape[1] != 1:
            return None
        if y.shape != y_hat.shape:
            return None
        eps = 1e-15
        m = y.shape[0]
        one = np.ones(y.shape[0]).reshape((-1, 1))
        if self.penality == 'l2':
            theta_0 = self.theta + 0  # copy
            theta_0[0][0] = 0
            return float(((- (1 / m) * (y.T.dot(np.log(y_hat + eps)) + (one - y).T.dot(np.log(one - y_hat + eps)))) + ((self.lambda_ / (2 * m)) * theta_0.T.dot(theta_0))).item())
        return float(((- (1 / m) * (y.T.dot(np.log(y_hat + eps)) + (one - y).T.dot(np.log(one - y_hat + eps))))).item())

    def reg_log_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop.
        The three arrays must have compatible shapes.
        Args:
            x: has to be a numpy.ndarray, a matrix of shape m * n.
            y: has to be a numpy.ndarray, a vector of shape m * 1.
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
            lambda_: has to be a float.
        Returns:
            A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
            None if x, y or theta are empty numpy.ndarray.
            None if x, y or theta does not share compatibles shapes.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray):
            return None
        if len(y.shape) < 2 or y.shape[0] < 1 or y.shape[1] < 1:
            return None
        if len(x.shape) < 2 or x.shape[0] != y.shape[0] or x.shape[1] < 1:
            return None
        if x.shape[1] != self.theta.shape[0] and x.shape[1] + 1 != self.theta.shape[0]:
            return None
        if self.theta.shape[1] != 1:
            return None
        m1 = 1 / x.shape[0]
        theta_0 = self.theta + 0  # copy
        theta_0[0][0] = 0
        if x.shape[1] != self.theta.shape[0]:
            x = np.hstack((np.ones((x.shape[0], 1)), x))  # Add intercept
        y_hat = self.sigmoid_(x.dot(self.theta))
        return m1 * (x.T.dot(y_hat - y) + (self.lambda_ * theta_0))

    def log_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible shapes.
        Args:
            x: has to be an numpy.ndarray, a matrix of shape m * n.
            y: has to be an numpy.ndarray, a vector of shape m * 1. 
        Returns:
            The gradient as a numpy.ndarray, a vector of shape n * 1, containg the result of the formula for all j.
            None if x or y are empty numpy.ndarray.
            None if x or y do not have compatible shapes.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        if len(x.shape) < 2 or x.shape[0] < 1 or x.shape[1] < 1:
            return None
        if len(y.shape) != 2 or y.shape[0] != x.shape[0] or y.shape[1] != 1:
            return None
        if x.shape[1] != self.theta.shape[0] and x.shape[1] + 1 != self.theta.shape[0]:
            return None
        m1 = 1 / x.shape[0]
        if x.shape[1] != self.theta.shape[0]:
            x = np.hstack((np.ones((x.shape[0], 1)), x))  # Add intercept
        return m1 * x.T.dot(self.sigmoid_(x.dot(self.theta)) - y)

    def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a matrix of dimension m * n: (number of training examples, number of features).
            y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        Returns:
            new_theta: numpy.ndarray, a vector of dimension (number of features + 1, 1).
            None if there is a matching shape problem.
            None if x or y is not of expected type.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        if len(x.shape) < 2 or x.shape[0] < 1 or x.shape[1] < 1:
            return None
        if len(y.shape) != 2 or y.shape[0] != x.shape[0] or y.shape[1] != 1:
            return None
        if x.shape[1] + 1 != self.theta.shape[0]:
            return None
        x_int = np.hstack((np.ones((x.shape[0], 1)), x))
        if self.penality == 'l2':
            for _ in range(self.max_iter):
                self.theta = self.theta - (self.alpha * self.reg_log_gradient(x_int, y))
        else:
            for _ in range(self.max_iter):
                self.theta = self.theta - (self.alpha * self.log_gradient(x_int, y))
        return self.theta
