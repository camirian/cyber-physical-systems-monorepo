import numpy as np
import pytest
from coursera_ml.course_2.neural_network import Sequential, Dense, ReLU, Sigmoid

def test_sequential_forward_pass():
    """Verify that the sequential model can perform a full forward pass."""
    # Simple XOR-like structure (2 inputs -> 4 hidden -> 1 output)
    model = Sequential([
        Dense(2, 4),
        ReLU(),
        Dense(4, 1),
        Sigmoid()
    ])
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output = model.predict(X)
    
    assert output.shape == (4, 1)
    assert np.all(output >= 0) and np.all(output <= 1)

def test_relu_activation():
    """Verify ReLU logic."""
    relu = ReLU()
    X = np.array([-1, 0, 1])
    expected = np.array([0, 0, 1])
    assert np.array_equal(relu.forward(X), expected)

def test_sigmoid_activation():
    """Verify Sigmoid logic."""
    sigmoid = Sigmoid()
    X = np.array([0])
    assert sigmoid.forward(X)[0] == 0.5
