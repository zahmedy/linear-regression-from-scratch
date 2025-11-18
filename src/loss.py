import numpy as np

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean Squared Error:
        L = mean((y_true - y_pred)^2)
    """
    return np.mean((y_pred-y_true) ** 2)

def mse_gradients(X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray):
    """
    Compute analytical gradients of MSE wrt w and b

    dL/dw = (2/N) * sum((y_pred - y_true) * X)
    dL/db = (2/N) * sum(y_pred - y_true)
    """
    N = X.shape[0]
    error = y_pred - y_true

    dL_dw = (2 / N) * np.sum(error * X)
    dL_db = (2 / N) * np.sum(error)

    return dL_dw, dL_db