import unittest
import numpy as np
from src.phase_4_mbse.simplex_manager import SimplexManager, HighPerformanceController, BaselineController
from src.phase_4_mbse.cbf_safety_filter import CBFSafetyFilter

class TestSimplexGuard(unittest.TestCase):
    def setUp(self):
        # 1D Kinematics for a robot approaching an obstacle
        self.x = np.array([1.0]) # 1.0m from wall
        self.safe_distance = 0.5
        self.f_x = np.array([0.0])
        self.g_x = np.array([[1.0]])
        self.grad_h = np.array([1.0]) # h(x) = x - safe_distance, grad_h = 1
        
        # Initialize controllers and safety monitor
        self.hpc = HighPerformanceController()
        self.bc = BaselineController()
        self.sm = CBFSafetyFilter(alpha=2.0)
        self.manager = SimplexManager(self.hpc, self.bc, self.sm)

    def test_safe_ai_control(self):
        """
        AI wants to move slowly away from the wall (safe).
        Control should remain with the AI.
        """
        # Override HPC to move safely
        class SafeHPC(HighPerformanceController):
            def get_action(self, x): return np.array([1.0])
        
        self.manager.hpc = SafeHPC()
        h_x = self.x[0] - self.safe_distance # 0.5m (Safe)
        
        u_final = self.manager.execute_step(self.x, self.f_x, self.g_x, h_x, self.grad_h)
        
        self.assertEqual(u_final[0], 1.0)
        self.assertFalse(self.manager.override_active)

    def test_unsafe_ai_override(self):
        """
        AI wants to move fast towards the wall (unsafe).
        Simplex should revoke authority and engage the Baseline Controller.
        """
        # HPC wants v = -5.0 (crash)
        h_x = self.x[0] - self.safe_distance # 0.5m (Safe, but AI is moving too fast)
        
        u_final = self.manager.execute_step(self.x, self.f_x, self.g_x, h_x, self.grad_h)
        
        # Final control should be from BC (v = 0.0)
        self.assertEqual(u_final[0], 0.0)
        self.assertTrue(self.manager.override_active)

if __name__ == '__main__':
    unittest.main()
