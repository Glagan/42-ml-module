import numpy as np
from my_linear_regression import MyLinearRegression


class MyRidge(MyLinearRegression):
    """
    My personnal ridge regression class to fit like a boss.
    fit_ and predict_ is not needed since it's the same as in the parent class.
    """

    def __init__(self, thetas: np.ndarray, alpha: float = 0.001, max_iter: int = 42000, lambda_: float = 0.5) -> None:
        super().__init__(thetas, alpha=alpha, max_iter=max_iter)
        if not isinstance(lambda_, float) and not isinstance(lambda_, int):
            raise TypeError('lambda_ should be a positive number')
        if lambda_ < 0:
            raise ValueError('lambda_ should be a positive number')
        self.lambda_ = lambda_

    def get_params_(self) -> dict:
        """
        Get all of the parameters used in the (Regularized) Linear Regression class.
        """
        return {
            "thetas": self.thetas,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "lambda_": self.lambda_,
        }

    def set_params_(self, value: dict) -> None:
        """
        Set all of the parameters used in the (Regularized) Linear Regression class.
        Silently ignore any errors.
        """
        if isinstance(value, dict):
            for key, value in value.items():
                if key == "lambda_":
                    if not isinstance(value, float) and not isinstance(value, int):
                        continue
                    if value < 0:
                        continue
                    setattr(self, key, value)
                elif key == "max_iter":
                    if not isinstance(value, int):
                        continue
                    if value < 10:
                        continue
                    setattr(self, key, value)
                elif key == "alpha":
                    if not isinstance(value, float):
                        continue
                    if value < 0:
                        continue
                    setattr(self, key, value)
                elif key == "thetas":
                    if not isinstance(value, np.ndarray) or len(value.shape) != 2 or value.shape[1] != 1:
                        continue
                    setattr(self, key, value)

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
        theta_0 = self.thetas + 0  # copy
        theta_0[0][0] = 0
        return np.square(y_hat - y) + self.lambda_ * theta_0.T.dot(theta_0)

    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop. The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
            lambda_: has to be a float.
        Returns:
            The regularized loss as a float.
            None if y, y_hat, or theta are empty numpy.ndarray.
            None if y and y_hat do not share the same shapes.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
            return None
        if len(y.shape) < 2 or y.shape[0] < 1 or y.shape[1] < 1 or y.shape != y_hat.shape:
            return None
        theta_0 = self.thetas + 0  # copy
        theta_0[0][0] = 0
        diff = (y_hat - y).T.dot(y_hat - y)
        return float(((1 / (2 * y.shape[0])) * (diff + self.lambda_ * theta_0.T.dot(theta_0))).item())

    def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the regularized linear gradient of three non-empty numpy.ndarray, without any for-loop.
        The three arrays must have compatible shapes.
        Args:
            y: has to be a numpy.ndarray, a vector of shape m * 1.
            x: has to be a numpy.ndarray, a matrix of dimesion m * n.
            theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
            lambda_: has to be a float.
        Return:
            A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
            None if y, x, or theta are empty numpy.ndarray.
            None if y, x or theta does not share compatibles shapes.
            None if y, x or theta or lambda_ is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray):
            return None
        if len(y.shape) < 2 or y.shape[0] < 1 or y.shape[1] < 1:
            return None
        if len(x.shape) < 2 or x.shape[0] != y.shape[0] or x.shape[1] < 1:
            return None
        if x.shape[1] + 1 != self.thetas.shape[0] and x.shape[1] != self.thetas.shape[0]:
            return None
        m1 = 1 / x.shape[0]
        theta_0 = self.thetas + 0  # copy
        theta_0[0][0] = 0
        if x.shape[1] != self.thetas.shape[0]:
            x = np.hstack((np.ones((x.shape[0], 1)), x))  # Add intercept if needed
        y_hat = x.dot(self.thetas)
        return m1 * (x.T.dot(y_hat - y) + (self.lambda_ * theta_0))

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
        if x.shape[1] + 1 != self.thetas.shape[0]:
            return None
        loss = []
        x_int = np.hstack((np.ones((x.shape[0], 1)), x))
        for _ in range(self.max_iter):
            self.thetas = self.thetas - (self.alpha * self.gradient_(x_int, y))
            if check_loss:
                loss.append(self.loss_(y, self.predict_(x_int)))
        return self.thetas, loss
