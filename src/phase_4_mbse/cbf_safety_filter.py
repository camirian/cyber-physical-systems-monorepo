import numpy as np
import scipy.optimize as opt

class CBFSafetyFilter:
    """
    Phase 4: MBSE - Control Barrier Function (CBF) Safety Filter
    
    This is the mathematical implementation of the "Blast Radius".
    No matter what output the AI (Phase 3) produces, this filter sits between
    the AI and the physical actuator (Phase 5).
    
    It solves a fast Quadratic Program (QP) to find the closest SAFE action
    to the AI's requested action. If the AI action is safe, it passes unmodified.
    If it violates the barrier function h(x) >= 0, it is minimally perturbed.
    
    System affine dynamics:
    x_dot = f(x) + g(x)u
    
    CBF Constraint:
    L_f h(x) + L_g h(x)u + alpha * h(x) >= 0
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def filter_action(self, x: np.ndarray, u_nom: np.ndarray, f_x: np.ndarray, g_x: np.ndarray, h_x: float, grad_h: np.ndarray) -> np.ndarray:
        """
        Solves the QP:
        min_u || u - u_nom ||^2
        s.t. grad_h * (f(x) + g(x)u) + alpha * h_x >= 0
        
        Args:
            x: Current physical state
            u_nom: Nominal control action requested by Neural Network
            f_x: Unforced dynamics f(x)
            g_x: Control matrix g(x)
            h_x: Evaluation of the barrier function h(x) (must be >= 0 for safety)
            grad_h: Gradient of h with respect to x
        """
        # Lie derivatives
        Lf_h = np.dot(grad_h, f_x)
        Lg_h = np.dot(grad_h, g_x)
        
        # We want: Lf_h + Lg_h * u + alpha * h_x >= 0
        # Re-arrange for standard QP constraint form: A*u <= b
        # -Lg_h * u <= Lf_h + alpha * h_x
        
        A_ineq = -Lg_h.reshape(1, -1)
        b_ineq = np.array([Lf_h + self.alpha * h_x])
        
        # Objective function: 1/2 * u^T * Q * u + c^T * u
        # We want to minimize ||u - u_nom||^2 = u^T u - 2 * u_nom^T * u + const
        dim_u = u_nom.shape[0]
        P = np.eye(dim_u)
        q = -u_nom
        
        # Solve using SLSQP (Sequential Least Squares Programming)
        def objective(u):
            return 0.5 * np.dot(u, u) - np.dot(u_nom, u)
        
        def constraint(u):
            # A_ineq * u <= b_ineq  =>  b_ineq - A_ineq * u >= 0
            return (b_ineq - np.dot(A_ineq, u))[0]
            
        cons = ({'type': 'ineq', 'fun': constraint})
        
        res = opt.minimize(objective, u_nom, method='SLSQP', constraints=cons)
        
        if res.success:
            return res.x
        else:
            # Fallback: Severe safety violation and QP failed. 
            # Engage hard mechanical rollback (zero energy state).
            print("[CRITICAL] CBF Filter QP Failed. Engaging Hardware Fallback.")
            return np.zeros_like(u_nom)

if __name__ == "__main__":
    print("--- Cyber-Physical CBF Safety Filter Initialized ---")
    
    # Mock Scenario: Robot approaching a wall. 
    # State x: [position] (1D kinematics for simplicity)
    x = np.array([1.0]) # 1 meter away from the wall
    
    # AI wants to move fast towards the wall (unsafe)
    u_nom = np.array([-5.0]) # v = -5 m/s
    
    # Dynamics: x_dot = u (Velocity control)
    f_x = np.array([0.0])
    g_x = np.array([[1.0]])
    
    # Barrier function: h(x) = x - safe_distance
    # Safe distance from wall is 0.5m. System is safe if h(x) >= 0
    safe_distance = 0.5
    h_x = x[0] - safe_distance # Currently 1.0 - 0.5 = 0.5 > 0 (Safe)
    
    # Gradient of h(x) w.r.t position
    grad_h = np.array([1.0])
    
    cbf = CBFSafetyFilter(alpha=2.0)
    u_safe = cbf.filter_action(x, u_nom, f_x, g_x, h_x, grad_h)
    
    print(f"AI Requested Control (Velocity): {u_nom[0]:.2f} m/s")
    print(f"CBF Projected Safe Control:      {u_safe[0]:.2f} m/s")
    print("Notice how the CBF gracefully overrides the AI to prevent an impending collision.")
