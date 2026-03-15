import numpy as np
import pytest
from coursera_ml.course_1.linear_regression import LinearRegression

def test_linear_regression_fit():
    """Verify that the model can learn a simple linear relationship."""
    # Data: y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])
    
    model = LinearRegression(learning_rate=0.1, iterations=1000)
    model.fit(X, y)
    
    # Check if weights and bias are close to true values (2 and 1)
    assert pytest.approx(model.weights[0], 0.1) == 2.0
    assert pytest.approx(model.bias, 0.1) == 1.0

def test_predict():
    """Test the predict method with unseen data."""
    X_train = np.array([[1], [2]])
    y_train = np.array([2, 4])
    
    model = LinearRegression(learning_rate=0.1, iterations=500)
    model.fit(X_train, y_train)
    
    X_test = np.array([[3], [4]])
    predictions = model.predict(X_test)
    
    # Expected: [6, 8]
    assert pytest.approx(predictions[0], 0.1) == 6.0
    assert pytest.approx(predictions[1], 0.1) == 8.0

def test_model_not_trained_error():
    """Verify that calling predict() before fit() raises an error."""
    model = LinearRegression()
    with pytest.raises(ValueError, match="Model has not been trained yet"):
        model.predict(np.array([[1]]))
