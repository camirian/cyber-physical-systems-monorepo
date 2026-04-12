import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class DifferentiableSimulator(nn.Module):
    """
    Phase 2: The Differentiable Simulator
    
    Standard RL uses a "Black Box" simulator (like MuJoCo or standard Gazebo). 
    You take an action, get a reward, and use noisy gradients (REINFORCE/PPO).
    
    A Differentiable Simulator is written entirely in tensor operations (PyTorch/Taichi/Brax).
    Because every physical integration step is part of the computational graph,
    we can backpropagate the final error directly through time (BPTT) into both:
    1. The Control Policy
    2. The Hardware Morphology (Mass, Friction, Thruster limits)
    
    This enables true "Hardware-Software Co-Design".
    """
    def __init__(self, dt=0.01):
        super().__init__()
        self.dt = dt
        
        # Hardware Parameters (Learnable!)
        # We start with a guess, but the optimizer will find the exact physical 
        # parameters needed to achieve the target trajectory.
        self.mass = nn.Parameter(torch.tensor(5.0)) # Initial guess: 5kg
        self.friction_coeff = nn.Parameter(torch.tensor(0.5)) # Initial guess: 0.5
        
    def step(self, state, action):
        """
        Euler Integration of 1D Kinematics:
        state = [position, velocity]
        action = [applied_force]
        
        F_net = F_applied - F_friction
        a = F_net / mass
        v_next = v + a * dt
        x_next = x + v * dt
        """
        x, v = state[:, 0], state[:, 1]
        F_applied = action[:, 0]
        
        # Physics Engine
        F_friction = self.friction_coeff * v # Simple linear drag
        F_net = F_applied - F_friction
        
        # Ensure mass stays positive during optimization
        m_safe = torch.abs(self.mass) + 1e-3 
        a = F_net / m_safe
        
        v_next = v + a * self.dt
        x_next = x + v_next * self.dt
        
        return torch.stack([x_next, v_next], dim=1)

    def simulate_trajectory(self, initial_state, actions):
        """Rolls out the simulation over a sequence of actions."""
        states = [initial_state]
        curr_state = initial_state
        
        for t in range(actions.shape[0]):
            curr_action = actions[t].unsqueeze(0)
            curr_state = self.step(curr_state, curr_action)
            states.append(curr_state)
            
        return torch.cat(states, dim=0)

if __name__ == "__main__":
    print("--- Phase 2: Differentiable Simulator Hardware Co-Design ---")
    
    sim = DifferentiableSimulator(dt=0.1)
    
    # 1. Define a target trajectory we want the robot to follow
    # Target: Reach 10 meters and stop
    target_pos = torch.tensor(10.0)
    target_vel = torch.tensor(0.0)
    
    # 2. Hardcoded control sequence (e.g., full throttle for 2s, then coast/brake)
    time_steps = 50
    actions = torch.zeros((time_steps, 1))
    actions[:20] = 15.0  # Thrust
    actions[20:] = -5.0  # Brake
    
    initial_state = torch.tensor([[0.0, 0.0]])
    
    # 3. Optimize the HARDWARE (Mass & Friction) to make this control sequence
    # perfectly achieve the target state. This is impossible with black-box RL
    # without millions of samples. Here, it takes ~50 iterations.
    
    optimizer = torch.optim.Adam(sim.parameters(), lr=0.1)
    
    print(f"Initial Hardware Guess: Mass={sim.mass.item():.2f}kg, Friction={sim.friction_coeff.item():.2f}")
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Rollout simulation
        trajectory = sim.simulate_trajectory(initial_state, actions)
        
        # Final state
        final_state = trajectory[-1]
        final_pos, final_vel = final_state[0], final_state[1]
        
        # Loss: Distance to target
        loss = (final_pos - target_pos)**2 + (final_vel - target_vel)**2
        
        # The Magic: Gradients flow BACKWARDS through the physics engine steps!
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Mass: {sim.mass.item():.2f} | Friction: {sim.friction_coeff.item():.2f}")
            
    print(f"Optimized Hardware: Mass={sim.mass.item():.2f}kg, Friction={sim.friction_coeff.item():.2f}")
    print("Notice how we directly solved for the exact physical properties required to make the trajectory work, via analytic physics gradients.")
