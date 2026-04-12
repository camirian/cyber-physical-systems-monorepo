import torch
import torch.nn as nn

class SO3EquivariantLayer(nn.Module):
    """
    Phase 1: The Epistemological Baseline - SO(3) Equivariance
    
    Standard Neural Networks (MLPs/CNNs) treat spatial coordinates (x, y, z) 
    as independent features. If the robot or sensor rotates, the network's 
    output changes unpredictably.
    
    This layer enforces SO(3) Equivariance: f(R * x) = R * f(x)
    where R is any 3D rotation matrix.
    
    Mechanism: 
    We extract rotation-INVARIANT features (like the vector norm ||x||), 
    pass those invariants through an arbitrary Neural Network to get a scalar, 
    and then multiply that scalar back onto the original rotation-EQUIVARIANT 
    vector. 
    
    This guarantees that the network respects the physical geometry of 3D space,
    making it immune to coordinate-frame transforms (e.g., IMU mounting offsets).
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        # The invariant pathway (processes magnitudes, not directions)
        self.invariant_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, v):
        """
        Args:
            v: Tensor of shape (batch_size, 3) representing 3D vectors (e.g., IMU accel)
        Returns:
            out_v: Equivariant transformed 3D vectors
        """
        # 1. Extract SO(3) Invariant feature (Euclidean Norm)
        # Adding epsilon to prevent NaN gradients at exactly 0
        v_norm = torch.norm(v, dim=1, keepdim=True) + 1e-8 
        
        # 2. Process invariant feature through MLP
        # This determines HOW MUCH to scale the vector based on its magnitude
        scalar_multiplier = self.invariant_mlp(v_norm)
        
        # 3. Re-modulate the original equivariant vector
        # Since v_norm is invariant, and scalar_multiplier is invariant,
        # multiplying them by the equivariant vector 'v' results in a purely
        # equivariant output.
        out_v = scalar_multiplier * v
        
        return out_v

if __name__ == "__main__":
    print("--- Phase 1: SO(3) Equivariance Mathematical Proof ---")
    
    # 1. Define a random 3D vector (e.g., gravity vector from IMU)
    v_original = torch.tensor([[1.0, 2.0, 3.0]])
    
    # 2. Define a 90-degree rotation matrix around the Z-axis
    theta = torch.tensor(torch.pi / 2)
    R_z = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0.0],
        [torch.sin(theta),  torch.cos(theta), 0.0],
        [0.0,              0.0,               1.0]
    ])
    
    # Rotate the input vector
    v_rotated = v_original @ R_z.T
    
    # 3. Initialize the Equivariant Layer
    layer = SO3EquivariantLayer()
    
    # 4. Process both the original and rotated vectors
    out_original = layer(v_original)
    out_rotated = layer(v_rotated)
    
    # 5. Proof: Rotating the output of the original vector should exactly equal
    # the output of the rotated input vector. f(R*x) == R*f(x)
    out_original_then_rotated = out_original @ R_z.T
    
    print(f"Input Vector (Original): {v_original.numpy()[0]}")
    print(f"Input Vector (Rotated):  {v_rotated.numpy()[0]}")
    print("-" * 50)
    print(f"f(R * x) [Output of rotated input]:     {out_rotated.detach().numpy()[0]}")
    print(f"R * f(x) [Rotated output of original]:  {out_original_then_rotated.detach().numpy()[0]}")
    
    # Assert numerical equivalence
    diff = torch.max(torch.abs(out_rotated - out_original_then_rotated)).item()
    print(f"Maximum Deviation: {diff:.8e}")
    if diff < 1e-6:
        print(">> PROOF SUCCESSFUL: Layer is mathematically SO(3) Equivariant.")
    else:
        print(">> PROOF FAILED: Equivariance broken.")
