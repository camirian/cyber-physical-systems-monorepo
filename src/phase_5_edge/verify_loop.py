import time
import random

class HardwareEdgeNode:
    """
    Phase 5: The Edge - Real-Time Hardware Execution
    
    This simulates the execution loop running on a Jetson Orin Nano or similar edge device.
    It enforces strict timing constraints and implements the "Wait-and-Verify" protocol.
    
    If the AI inference takes too long (violating the latency budget), or if the
    leading metric predicts a failure, the hardware immediately cuts power to the
    actuators (Graceful Degradation).
    """
    def __init__(self, latency_budget_ms=10.0):
        self.latency_budget_ms = latency_budget_ms
        self.system_armed = True
        
    def read_sensors(self):
        """Simulate reading high-frequency IMU/Encoder data."""
        # Returns [position, velocity]
        return {"pos": random.uniform(0, 5), "vel": random.uniform(-2, 2)}

    def run_ai_inference(self, sensor_data):
        """
        Simulates AI inference (e.g., via TensorRT).
        Randomly injects latency to simulate thermal throttling or OS jitter.
        """
        start_time = time.perf_counter()
        
        # Simulate computation time
        compute_time = random.uniform(2, 15) / 1000.0 # 2ms to 15ms
        time.sleep(compute_time)
        
        # Simulated nominal control output
        ai_action = -1.5 * sensor_data["vel"] # Simple damping response
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return ai_action, elapsed_ms

    def WaitAndVerify_Loop(self):
        """
        The critical execution loop.
        """
        print(f"--- Edge Node Armed. Latency Budget: {self.latency_budget_ms}ms ---")
        
        for tick in range(1, 6):
            if not self.system_armed:
                print("System disarmed. Requires manual reset.")
                break
                
            print(f"\n[Tick {tick}]")
            
            # 1. Read
            data = self.read_sensors()
            
            # 2. Infer
            action, inference_time_ms = self.run_ai_inference(data)
            
            # 3. VERIFY (The Edge Safety Check)
            if inference_time_ms > self.latency_budget_ms:
                print(f"[FATAL] Latency Violation: {inference_time_ms:.2f}ms > {self.latency_budget_ms}ms")
                print(">>> TRIGGERING HARDWARE ROLLBACK. CUTTING MOTOR POWER. <<<")
                self.system_armed = False
                action = 0.0 # Zero energy state
            else:
                print(f"[OK] Inference Latency: {inference_time_ms:.2f}ms")
                print(f"Actuating Motor Torque: {action:.2f} Nm")

if __name__ == "__main__":
    random.seed(42) # For reproducible "jitter"
    edge_node = HardwareEdgeNode(latency_budget_ms=10.0)
    edge_node.WaitAndVerify_Loop()
