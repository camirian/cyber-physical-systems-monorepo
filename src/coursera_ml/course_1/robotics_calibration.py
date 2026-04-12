import numpy as np
import matplotlib.pyplot as plt
from coursera_ml.course_1.linear_regression import LinearRegression

def generate_sensor_data(n_samples: int = 100):
    """
    Simulates a ToF (Time-of-Flight) sensor with a linear calibration error.
    
    In reality, sensors often have a 'systematic error': 
    Measured = (Scale * True_Distance) + Offset + Noise
    """
    np.random.seed(42)
    true_distances = np.linspace(0.1, 2.0, n_samples).reshape(-1, 1) # 10cm to 2m
    
    # Real-world systematic error: scale=1.05 (5% error), offset=0.03 (3cm)
    scale = 1.05
    offset = 0.03
    noise = np.random.normal(0, 0.01, (n_samples, 1)) # 1cm Gaussian noise
    
    measured_distances = (scale * true_distances) + offset + noise
    return true_distances, measured_distances

def run_calibration_example():
    print("--- Robotics Sensor Calibration Example ---")
    print("Scenario: Calibrating a ToF sensor with systematic scale and offset errors.")
    
    # 1. Generate 'Ground Truth' vs 'Measured' data
    X_true, y_measured = generate_sensor_data()
    
    # 2. Train Linear Regression to find the mapping: Measured -> True
    # We want to find the inverse mapping to 'fix' the sensor
    model = LinearRegression(learning_rate=0.5, iterations=1500)
    model.fit(y_measured, X_true.flatten())
    
    print(f"Calibration Complete.")
    print(f"Detected Inverse Scale (Weights): {model.weights[0]:.4f}")
    print(f"Detected Inverse Offset (Bias): {model.bias:.4f}")
    
    # 3. Test on a new measurement
    raw_reading = np.array([[1.5]]) # Sensor says 1.5m
    calibrated_reading = model.predict(raw_reading)
    print(f"Raw Reading: {raw_reading[0][0]}m -> Calibrated: {calibrated_reading[0]:.4f}m")

    # 4. Optional: Save a plot to show the '10x better' visual approach
    plt.figure(figsize=(10, 6))
    plt.scatter(y_measured, X_true, alpha=0.5, label='Noisy Measurements')
    plt.plot(y_measured, model.predict(y_measured), color='red', label='Calibration Curve')
    plt.xlabel('Measured Distance (m)')
    plt.ylabel('Ground Truth (m)')
    plt.title('ToF Sensor Calibration via Linear Regression')
    plt.legend()
    plt.grid(True)
    # Note: In a real CLI environment we might save to disk
    # plt.savefig('sensor_calibration.png')
    print("Calibration curve calculated successfully.")

if __name__ == "__main__":
    run_calibration_example()
