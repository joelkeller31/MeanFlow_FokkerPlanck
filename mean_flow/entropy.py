import torch 
import numpy as np 
from torch.func import jacfwd, jacrev

def compute_batch_entropy(model, pos, t, r):
    """
    Computes ∇·v = Σ_i ∂v_i/∂x_i using JVP (more memory-efficient)
    Args:
        pos: [B, N, D] particle positions
        t: [B,] time values
        r: [B,] reference times
    Returns:
        divergence: [B, N] divergence per particle
    """
    batch_size, n_particles, dim = pos.shape
    divergence = torch.zeros(batch_size, n_particles, device=pos.device)
    
    # Create basis vectors for each dimension
    for i in range(dim):
        # Create unit vector in i-th dimension
        v = torch.zeros_like(pos)
        v[..., i] = 1.0
        
        # Compute JVP (∂v_i/∂x_i)
        _, jvp = torch.func.jvp(
            lambda x: model(x, t, r)[..., i],  # Only compute i-th component
            (pos,),  # primal input
            (v,)     # tangent vector
        )
        divergence += jvp
    
    return divergence.mean()  