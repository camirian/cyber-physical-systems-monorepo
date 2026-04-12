import numpy as np
from src.phase_4_mbse.cbf_safety_filter import CBFSafetyFilter

class HighPerformanceController:
    """
    HPC: Represents the complex, unverified AI/Neural Network.
    In a real scenario, this would be a trained PyTorch model or RL agent.
    """
    def get_action(self, x: np.ndarray) -> np.ndarray:
        # Mock: AI wants to move fast (e.g., towards an obstacle)
        return np.array([-5.0])

class BaselineController:
    """
    BC: Represents the mathematically verified, safe, but sub-optimal controller.
    e.g., a PID controller that keeps the system at a standstill or low velocity.
    """
    def get_action(self, x: np.ndarray) -> np.ndarray:
        # Mock: Baseline keeps the robot still (safe)
        return np.array([0.0])

class SimplexManager:
    """
    Phase 4: MBSE - Adaptive Simplex Architecture Manager
    
    This class orchestrates the "Guard" logic:
    1. Receives an action from the High-Performance Controller (HPC).
    2. Passes the action through the Safety Monitor (SM) / CBF Filter.
    3. If the SM detects a significant divergence or failure, it overrides
       the HPC and switches control to the Baseline Controller (BC).
    """
    def __init__(self, hpc: HighPerformanceController, bc: BaselineController, sm: CBFSafetyFilter):
        self.hpc = hpc
        self.bc = bc
        self.sm = sm
        self.override_active = False

    def execute_step(self, x: np.ndarray, f_x: np.ndarray, g_x: np.ndarray, h_x: float, grad_h: np.ndarray) -> np.ndarray:
        """
        Executes a single control step, guarding the physical system.
        """
        # 1. Get nominal action from the "untrusted" AI
        u_hpc = self.hpc.get_action(x)
        
        # 2. Pass through Safety Monitor (CBF Filter)
        # The filter solves a QP to find the closest safe action.
        u_safe = self.sm.filter_action(x, u_hpc, f_x, g_x, h_x, grad_h)
        
        # 3. Logic: If the 'safe' action is significantly different from the HPC action,
        # or if the QP failed (resulting in a zero-action rollback), we flag a safety override.
        divergence = np.linalg.norm(u_safe - u_hpc)
        
        if divergence > 1e-3:
            if not self.override_active:
                print(f"[SIMPLEX] Safety Boundary Breach Detected (Divergence: {divergence:.4f}).")
                print("[SIMPLEX] Revoking HPC Authority. Engaging Baseline Controller.")
                self.override_active = True
            
            # Use Baseline Controller for the final output to ensure stability
            return self.bc.get_action(x)
        else:
            self.override_active = False
            return u_safe

if __name__ == "__main__":
    print("--- Simplex Safety Manager Initialized ---")
    
    # Setup: Robot approaching a wall
    x = np.array([1.0]) # 1.0m from wall
    safe_distance = 0.5
    h_x = x[0] - safe_distance
    grad_h = np.array([1.0])
    f_x = np.array([0.0])
    g_x = np.array([[1.0]])
    
    # Initialize components
    hpc = HighPerformanceController()
    bc = BaselineController()
    sm = CBFSafetyFilter(alpha=2.0)
    
    manager = SimplexManager(hpc, bc, sm)
    
    # Simulate step
    u_final = manager.execute_step(x, f_x, g_x, h_x, grad_h)
    
    print(f"Final Command to Hardware: {u_final[0]:.2f} m/s")
    if manager.override_active:
        print("STATUS: SYSTEM GUARDED BY BASELINE CONTROLLER.")
    else:
        print("STATUS: SYSTEM OPERATING UNDER AI CONTROL.")
