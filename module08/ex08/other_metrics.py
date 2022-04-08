import numpy as np


def confusion_matrix_(y: np.ndarray, y_hat: np.ndarray, pos_label=1):
    """
    Compute a confusion matrix [TP, TN, FP, FN] from 2 numpy.ndarray
    Args:
        y: a numpy.ndarray for the correct labels
        y_hat: a numpy.ndarray for the predicted labels
    Returns:
        The confusion matrix as a list of float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if len(y.shape) < 1 or any([True if size < 1 else False for size in y.shape]):
        return None
    if y.shape != y_hat.shape:
        return None
    if not isinstance(pos_label, int) and not isinstance(pos_label, str):
        return None
    if isinstance(pos_label, int) and pos_label < 0:
        return None
    return [
        np.sum((y == pos_label) & (y_hat == pos_label)),
        np.sum((y != pos_label) & (y_hat != pos_label)),
        np.sum((y != pos_label) & (y_hat == pos_label)),
        np.sum((y == pos_label) & (y_hat != pos_label)),
    ]


def accuracy_score_(y: np.ndarray, y_hat: np.ndarray):
    """
    Compute the accuracy score.
    Reference:
        https://parasite.id/blog/2018-12-13-model-evaluation/
    Args:
        y: a numpy.ndarray for the correct labels
        y_hat: a numpy.ndarray for the predicted labels
    Returns:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if len(y.shape) < 1 or any([True if size < 1 else False for size in y.shape]):
        return None
    if y.shape != y_hat.shape:
        return None
    return np.sum(y == y_hat) / y.shape[0]


def precision_score_(y: np.ndarray, y_hat: np.ndarray, pos_label=1):
    """
    Compute the precision score.
    Args:
        y: a numpy.ndarray for the correct labels
        y_hat: a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if len(y.shape) < 1 or any([True if size < 1 else False for size in y.shape]):
        return None
    if y.shape != y_hat.shape:
        return None
    if not isinstance(pos_label, int) and not isinstance(pos_label, str):
        return None
    if isinstance(pos_label, int) and pos_label < 0:
        return None
    tp, tn, fp, fn = confusion_matrix_(y, y_hat, pos_label)
    return (tp) / (tp + fp)


def recall_score_(y: np.ndarray, y_hat: np.ndarray, pos_label=1):
    """
    Compute the recall score.
    Args:
        y: a numpy.ndarray for the correct labels
        y_hat: a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if len(y.shape) < 1 or any([True if size < 1 else False for size in y.shape]):
        return None
    if y.shape != y_hat.shape:
        return None
    if not isinstance(pos_label, int) and not isinstance(pos_label, str):
        return None
    if isinstance(pos_label, int) and pos_label < 0:
        return None
    tp, tn, fp, fn = confusion_matrix_(y, y_hat, pos_label)
    return (tp) / (tp + fn)


def f1_score_(y: np.ndarray, y_hat: np.ndarray, pos_label=1):
    """
    Compute the f1 score.
    Args:
        y: a numpy.ndarray for the correct labels
        y_hat: a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if len(y.shape) < 1 or any([True if size < 1 else False for size in y.shape]):
        return None
    if y.shape != y_hat.shape:
        return None
    if not isinstance(pos_label, int) and not isinstance(pos_label, str):
        return None
    if isinstance(pos_label, int) and pos_label < 0:
        return None
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)
