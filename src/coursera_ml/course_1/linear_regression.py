import numpy as np
from typing import Tuple, Dict, Any

class LinearRegression:
    """
    Production-grade implementation of Linear Regression with Gradient Descent.
    
    This implementation uses vectorized operations for efficiency, mirroring
    real-world ML library design.
    """
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.history: Dict[str, Any] = {"cost": []}

    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Computes the Mean Squared Error (MSE)."""
        m = len(y)
        predictions = X.dot(self.weights) + self.bias
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return float(cost)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fits the model to the training data using Gradient Descent.
        
        Args:
            X: Input features of shape (m, n)
            y: Target values of shape (m,)
        """
        m, n = X.shape
        # Initialize parameters
        self.weights = np.zeros(n)
        self.bias = 0.0

        for i in range(self.iterations):
            # Vectorized prediction
            predictions = X.dot(self.weights) + self.bias
            
            # Vectorized gradient calculation
            dw = (1 / m) * X.T.dot(predictions - y)
            db = (1 / m) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Log cost for monitoring
            if i % 100 == 0 or i == self.iterations - 1:
                cost = self._compute_cost(X, y)
                self.history["cost"].append(cost)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts target values for new input data."""
        if self.weights is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return X.dot(self.weights) + self.bias
