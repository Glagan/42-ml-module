import numpy as np


class MyLinearRegression:
    def __init__(self, theta: np.ndarray, alpha: float = 0.0005, max_iter: int = 42000) -> None:
        if not isinstance(theta, np.ndarray) or len(theta.shape) != 2 or theta.shape[1] != 1:
            raise ValueError(
                'theta should be an array of at size (n,1) with n > 1')
        if (not isinstance(alpha, float) and not isinstance(alpha, int)) or alpha < 0:
            raise TypeError('alpha should be a positive number')
        if (not isinstance(max_iter, float) and not isinstance(max_iter, int)) or max_iter < 1:
            raise TypeError('max_iter should be a positive number')
        self.theta = theta
        self.alpha = alpha
        self.max_iter = max_iter

    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have the compatible dimensions.
        Args:
            x: has to be an numpy.ndarray, a matrix of dimension m * n.
            y: has to be an numpy.ndarray, a vector of dimension m * 1.
        Returns:
            The gradient as a numpy.ndarray, a vector of dimensions n * 1, containg the result of the formula for all j.
            None if x, y, or theta are empty numpy.ndarray.
            None if x, y and theta do not have compatible dimensions.
            None if x, y or theta is not of expected type.
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
        m1 = 1 / (2 * x.shape[0])
        if x.shape[1] != self.theta.shape[0]:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        return m1 * x.T.dot(x.dot(self.theta) - y)

    def fit_(self, x: np.ndarray, y: np.ndarray, check_loss: bool = False) -> np.ndarray:
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
        loss = []
        x_int = np.hstack((np.ones((x.shape[0], 1)), x))
        for _ in range(self.max_iter):
            self.theta = self.theta - (self.alpha * self.gradient(x_int, y))
            if check_loss:
                loss.append(self.loss_(y, self.predict_(x_int)))
        return self.theta, loss

    def predict_(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the prediction vector y_hat from two non-empty numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a matrix of dimension m * n.
        Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray):
            return None
        if len(x.shape) != 2 or (x.shape[1] != self.theta.shape[0] and x.shape[1] + 1 != self.theta.shape[0]):
            return None
        if x.shape[1] != self.theta.shape[0]:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x.dot(self.theta)

    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """ 
        Calculates all the elements (y - x)^2.
        Args:
            x: has to be an numpy.ndarray, a vector.
            y: has to be an numpy.ndarray, a vector.
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
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if len(y.shape) < 2 or y.shape[0] < 1 or y.shape[1] != 1:
            return None
        if y.shape != y_hat.shape:
            return None
        return (1 / (2 * y.shape[0])) * np.sum(np.square(y_hat - y))
