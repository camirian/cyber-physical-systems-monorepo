import numpy as np
from abc import ABC, abstractmethod
from typing import List

class Layer(ABC):
    """Abstract Base Class for a Neural Network Layer."""
    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        pass

class Dense(Layer):
    """
    Fully Connected (Dense) Layer.
    
    Implements: Output = Activation(X.W + b)
    """
    def __init__(self, input_size: int, output_size: int):
        # He initialization for weights
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        return np.dot(input_data, self.weights) + self.bias

class ReLU(Layer):
    """Rectified Linear Unit Activation Layer."""
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        return np.maximum(0, input_data)

class Sigmoid(Layer):
    """Sigmoid Activation Layer."""
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input_data))

class Sequential:
    """
    A Sequential Container for Layers.
    
    Mirrors the API of high-level frameworks like Keras/PyTorch.
    """
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
