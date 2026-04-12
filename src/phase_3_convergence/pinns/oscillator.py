import torch
import torch.nn as nn
import numpy as np

class HarmonicOscillatorPINN(nn.Module):
    """
    Phase 3: The Physics-AI Convergence
    Physics-Informed Neural Network (PINN) for a Simple Harmonic Oscillator.
    
    Instead of merely fitting (t, x) data, this network computes the physical
    residual (the differential equation) and penalizes violations of physics.
    
    Equation: m * x''(t) + mu * x'(t) + k * x(t) = 0
    """
    def __init__(self, m=1.0, mu=0.1, k=2.0):
        super().__init__()
        # Physical Constants
        self.m = m    # Mass
        self.mu = mu  # Damping coefficient
        self.k = k    # Spring constant
        
        # Neural Network Architecture (t -> x)
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(), # Tanh provides smooth, continuous derivatives necessary for PINNs
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        """Forward pass: Time -> Displacement"""
        return self.net(t)

    def compute_physics_residual(self, t):
        """
        Computes the deviation from the governing differential equation.
        Requires Autograd to compute d^2x/dt^2 and dx/dt.
        """
        t.requires_grad_(True)
        
        # 1. Forward pass
        x = self.forward(t)
        
        # 2. Compute first derivative: dx/dt (Velocity)
        dx_dt = torch.autograd.grad(
            x, t, grad_outputs=torch.ones_like(x),
            create_graph=True, retain_graph=True
        )[0]
        
        # 3. Compute second derivative: d^2x/dt^2 (Acceleration)
        d2x_dt2 = torch.autograd.grad(
            dx_dt, t, grad_outputs=torch.ones_like(dx_dt),
            create_graph=True, retain_graph=True
        )[0]
        
        # 4. Calculate the physics residual based on the equation of motion
        residual = (self.m * d2x_dt2) + (self.mu * dx_dt) + (self.k * x)
        return residual

    def loss_function(self, t_data, x_data, t_physics, lambda_phys=1e-4):
        """
        L_total = L_data + lambda * L_physics
        """
        # 1. Data Loss (MSE against actual sensor telemetry)
        x_pred = self.forward(t_data)
        loss_data = nn.MSELoss()(x_pred, x_data)
        
        # 2. Physics Loss (MSE of the differential equation residual against 0)
        residual = self.compute_physics_residual(t_physics)
        loss_physics = nn.MSELoss()(residual, torch.zeros_like(residual))
        
        # 3. Total Loss
        total_loss = loss_data + (lambda_phys * loss_physics)
        
        return total_loss, loss_data, loss_physics

if __name__ == "__main__":
    # Smoke test the architecture
    pinn = HarmonicOscillatorPINN()
    t_test = torch.linspace(0, 10, 100).view(-1, 1)
    x_test = torch.sin(t_test) # Dummy data
    
    loss, l_data, l_phys = pinn.loss_function(t_test, x_test, t_test)
    print("--- Cyber-Physical PINN Initialized ---")
    print(f"Total Loss: {loss.item():.4f} | Data Loss: {l_data.item():.4f} | Physics Residual Loss: {l_phys.item():.4f}")
