import numpy as np


class MyLinearRegression:
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas: np.ndarray, alpha=0.001, max_iter=100000):
        if isinstance(thetas, list):
            thetas = np.array(thetas)
        if not isinstance(thetas, np.ndarray) or len(thetas.shape) != 2 or thetas.shape[0] < 1 or thetas.shape[1] != 1:
            raise ValueError("thetas should be a non-empty row Vector")
        if not isinstance(alpha, float) or not alpha or alpha < 0:
            raise ValueError("alpha should be a positive float")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter should be a positive number")
        self.thetas = thetas
        self.alpha = alpha
        self.max_iter = max_iter

    def gradient_(self, x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Computes a gradient vector from three non-empty numpy.ndarray, without any for loop. The three arrays must have compatible dimensions.
        Args:
            x: has to be a numpy.ndarray, a matrix of dimension m * 2.
            y: has to be a numpy.ndarray, a vector of dimension m * 1.
            theta: has to be a numpy.ndarray, a 2 * 1 vector.
        Returns:
            The gradient as a numpy.array, a vector of shape 2 * 1.
            None if x, y, or theta is an empty numpy.array.
            None if x, y and theta do not have compatible shapes.
            None if x, y or theta is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or len(x.shape) != 2 or x.shape[0] < 1 or x.shape[1] != 2:
            return None
        if not isinstance(y, np.ndarray) or len(y.shape) != 2 or y.shape[0] != x.shape[0] or y.shape[1] != 1:
            return None
        if not isinstance(theta, np.ndarray) or theta.shape != (2, 1):
            return None
        m1 = 1 / x.shape[0]
        return (m1 * (x.T.dot(x.dot(theta) - y)))

    def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
            y: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
        Return:
            new_theta: numpy.array, a vector of shape 2 * 1.
            None if there is a matching shape problem.
            None if x or y is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or len(x.shape) != 2 or x.shape[0] < 1 or x.shape[1] != 1:
            return None
        if not isinstance(y, np.ndarray) or y.shape != x.shape:
            return None
        if self.thetas.shape[0] != x.shape[1] + 1:
            return None
        if self.max_iter < 1:
            return None
        x = np.hstack((np.ones((x.shape[0], 1)), x))  # Add intercept
        for _ in range(self.max_iter):
            use_thetas = self.thetas
            self.thetas = (use_thetas - (self.alpha *
                           self.gradient_(x, y, use_thetas)))
        return self.thetas

    def predict_(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the vector of prediction y_hat from a non-empty numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a vector of dimension m * 1. 
        Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1. 
        Raises:
            This function should not raise any Exceptions.
        """
        if not isinstance(x, np.ndarray) or len(x.shape) != 2 or x.shape[0] < 1:
            return None
        if self.thetas.shape[0] != x.shape[1] + 1:
            return None
        with_intercept = np.hstack((np.ones((x.shape[0], 1)), x))
        return np.dot(with_intercept, self.thetas)

    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """
        Description:
            Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_elem: numpy.array, a vector of dimension (number of the training examples,1).
            None if there is a dimension matching problem between y and y_hat.
            None if y or y_hat is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray):
            return None
        if not isinstance(y_hat, np.ndarray):
            return None
        if len(y.shape) != 2 or y.shape[0] < 1 or y.shape != y_hat.shape:
            return None
        return np.square(y_hat - y)

    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Computes the half mean squared error of two non-empty numpy.array, without any for loop.
        The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            The half mean squared error of the two vectors as a float.
            None if y or y_hat are empty numpy.array.
            None if y and y_hat does not share the same dimensions.
            None if y or y_hat is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray):
            return None
        if not isinstance(y_hat, np.ndarray):
            return None
        if len(y.shape) != 2 or y.shape[0] < 1 or y.shape != y_hat.shape:
            return None
        return (1 / (2 * y.shape[0])) * np.sum(np.square(y_hat - y))
