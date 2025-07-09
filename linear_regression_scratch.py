import numpy as np

def predict(X, w, b):
    """
    Compute linear regression predictions.
    X: numpy array of shape (num_samples, num_features)
    w: numpy array of shape (num_features,)
    b: scalar (float)
    Returns: numpy array of predictions (num_samples,)
    """
    # Compute the dot product of X and w (weighted sum for each sample)
    # Add the bias term b to each prediction
    return np.dot(X, w) + b

def mean_squared_error(y_true, y_pred):
    """
    Compute the mean squared error between true and predicted values.
    y_true: numpy array of shape (num_samples,)
    y_pred: numpy array of shape (num_samples,)
    Returns: scalar (float)
    """
    return np.mean((y_true - y_pred) ** 2)
