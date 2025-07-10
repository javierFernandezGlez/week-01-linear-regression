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

def gradient_descent_step(X, y, w, b, learning_rate):
    """
    Perform one step of gradient descent for linear regression.
    X: (num_samples, num_features)
    y: (num_samples,)
    w: (num_features,)
    b: scalar
    learning_rate: float
    Returns: updated w, b
    """
    m = X.shape[0]  # number of samples
    y_pred = predict(X, w, b)
    error = y_pred - y

    # Compute gradients
    grad_w = (2/m) * np.dot(X.T, error)
    grad_b = (2/m) * np.sum(error)

    # Update parameters
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    return w, b

def train_linear_regression(X, y, learning_rate=0.01, epochs=1000, verbose=True):
    """
    Train linear regression using gradient descent.
    X: (num_samples, num_features)
    y: (num_samples,)
    learning_rate: float
    epochs: int, number of iterations
    verbose: bool, whether to print loss during training
    Returns: learned w, b, and list of loss values
    """
    num_features = X.shape[1]
    w = np.zeros(num_features)
    b = 0.0
    loss_history = []

    for epoch in range(epochs):
        y_pred = predict(X, w, b)
        loss = mean_squared_error(y, y_pred)
        loss_history.append(loss)
        w, b = gradient_descent_step(X, y, w, b, learning_rate)
        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return w, b, loss_history
