import numpy as np
from typing import Dict, Any

class LogisticRegression:
    """
    Production-grade Binary Logistic Regression implementation.
    
    Ideal for binary classification tasks like:
    - Motor Fault Detection (Success/Failure)
    - Sensor Health Monitoring (Normal/Faulty)
    """
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.history: Dict[str, Any] = {"cost": []}

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Computes the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def _compute_cost(self, X: np.ndarray, y: np.ndarray, predictions: np.ndarray) -> float:
        """Computes the Binary Cross-Entropy loss."""
        m = len(y)
        # Add epsilon to prevent log(0)
        epsilon = 1e-15
        cost = -(1 / m) * np.sum(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
        return float(cost)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fits the model to binary classification data."""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0

        for i in range(self.iterations):
            linear_model = X.dot(self.weights) + self.bias
            predictions = self._sigmoid(linear_model)
            
            # Gradients
            dw = (1 / m) * X.T.dot(predictions - y)
            db = (1 / m) * np.sum(predictions - y)

            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 100 == 0 or i == self.iterations - 1:
                cost = self._compute_cost(X, y, predictions)
                self.history["cost"].append(cost)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns the probability of the positive class (y=1)."""
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")
        linear_model = X.dot(self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions (0 or 1) based on a threshold."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
