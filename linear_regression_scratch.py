import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

print("Loading California Housing dataset...")
housing = fetch_california_housing()
X = housing.data  
y = housing.target  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

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

print("\n" + "="*50)
print("TRAINING LINEAR REGRESSION FROM SCRATCH")
print("="*50)

learning_rate = 0.01
epochs = 1000

w_scratch, b_scratch, loss_history = train_linear_regression(
    X_train_scaled, y_train, learning_rate=learning_rate, epochs=epochs, verbose=True
)

print(f"\nFinal training loss: {loss_history[-1]:.4f}")


# PLOT THE LOSS CURVE

plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Training Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 6. PREDICTED VS ACTUAL PLOT (SCRATCH MODEL) ---
print("\n" + "="*50)
print("PREDICTED VS ACTUAL VALUES (SCRATCH MODEL)")
print("="*50)

# Make predictions on the test set
y_pred_scratch = predict(X_test_scaled, w_scratch, b_scratch)

# Scatter plot: actual values on x-axis, predicted values on y-axis
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_scratch, alpha=0.5, color='green', label='Predicted vs Actual (Scratch)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction (y = x)')
plt.xlabel('Actual House Value (in $100,000s)', fontsize=12)
plt.ylabel('Predicted House Value (in $100,000s)', fontsize=12)
plt.title('Predicted vs Actual House Values (Scratch Model)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scratch_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nIf the model were perfect, all points would fall on the dashed red line. The closer the points are to this line, the better the predictions!")
