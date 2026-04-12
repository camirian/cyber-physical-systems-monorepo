import numpy as np
import pytest
from coursera_ml.course_1.logistic_regression import LogisticRegression

def test_logistic_regression_fit():
    """Verify classification on a simple linearly separable dataset."""
    # Data: Classes clearly separated by x=5
    X = np.array([[1], [2], [8], [9]])
    y = np.array([0, 0, 1, 1])
    
    model = LogisticRegression(learning_rate=1.0, iterations=1000)
    model.fit(X, y)
    
    # Predict probabilities
    probs = model.predict_proba(np.array([[3], [7]]))
    assert probs[0] < 0.5
    assert probs[1] > 0.5
    
    # Predict classes
    preds = model.predict(np.array([[3], [7]]))
    assert preds[0] == 0
    assert preds[1] == 1

def test_sigmoid_range():
    """Ensure sigmoid always returns values between 0 and 1."""
    model = LogisticRegression()
    large_values = np.array([-100.0, 0.0, 100.0])
    outputs = model._sigmoid(large_values)
    
    assert np.all(outputs >= 0)
    assert np.all(outputs <= 1)
    assert outputs[0] < 1e-10
    assert outputs[1] == 0.5
    assert outputs[2] > 1 - 1e-10
