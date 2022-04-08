import numpy as np
import pandas as pd


def confusion_matrix_(y_true: np.ndarray, y_hat: np.ndarray, labels: list = None, df_option: bool = False) -> np.ndarray:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true: a numpy.ndarray for the correct labels
        y_hat: a numpy.ndarray for the predicted labels
        labels: optional, a list of labels to index the matrix. This may be used to reorder or select a subset of labels. (default=None)
        df_option: optional, if set to True the function will return a pandas DataFrame instead of a numpy array. (default=False)
    Returns:
        The confusion matrix as a numpy ndarray or a pandas DataFrame according to df_option value.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if len(y_true.shape) < 1 or any([True if size < 1 else False for size in y_true.shape]):
        return None
    if y_true.shape != y_hat.shape:
        return None
    all_labels = np.unique(np.stack((y_true, y_hat)))
    all_labels.sort()
    if labels is None:
        labels = all_labels.tolist()
    elif any([label not in all_labels for label in labels]):
        return None
    if not isinstance(labels, list):
        return None
    if not isinstance(df_option, bool):
        return None
    confusion_matrix = []
    for label_row in labels:
        matrix_row = []
        for label_col in labels:
            # Diagonal is True Positive
            if label_row == label_col:
                matrix_row.append(np.sum((y_true == label_row) & (y_hat == label_row)))
            # The other columns are the False Positive above the diagonal
            # and False Negative below the diagonal
            else:
                matrix_row.append(np.sum((y_true == label_row) & (y_hat == label_col)))
        confusion_matrix.append(matrix_row)
    if df_option:
        return pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    return np.array(confusion_matrix)
