### here's what i want 
import numpy as np
from typing import Callable, Tuple, Union




def generate_training_data(
    n_steps: int,
    dt: float,
    N: int,
    d: int,
    D_sqrt: float,
    x0: np.ndarray,  # Initial positions, shape (N, d)
    forcing: Callable[[np.ndarray, float], np.ndarray],
    rng: np.random.Generator, 
    Noisy: bool
) -> np.ndarray:
    
    ts = np.linspace(0, 1, n_steps + 1)
    trajs = np.zeros((n_steps + 1, N, d + 1))  
    
    trajs[0, :, :d] = x0  
    trajs[0, :, d] = ts[0]  
    
    for i in range(n_steps):
        # Update positions
        new_positions = euler_maruyama_step(
            particle_pos=trajs[i, :, :d],
            t=ts[i],
            D_sqrt=D_sqrt,
            dt=dt,
            rng=rng,
            noisy=Noisy,
            forcing=forcing
        )
        
        trajs[i+1, :, :d] = new_positions
        trajs[i+1, :, d] = ts[i+1]

    return trajs, ts # (shape n_points, N, d+1)

def euler_maruyama_step(
    particle_pos: np.ndarray,  
    t: float,
    D_sqrt: float,
    dt: float,
    forcing: Callable[[np.ndarray, float], np.ndarray],
    rng: np.random.Generator,
    noisy: bool = False, 
) -> np.ndarray:
    
    if noisy:

        noise = rng.normal(size=particle_pos.shape)
        diffusion = np.sqrt(2 * dt) * D_sqrt * noise
        return particle_pos + dt * forcing(particle_pos, t).reshape(particle_pos.shape) + diffusion
    else:
        return particle_pos + dt * forcing(particle_pos, t).reshape(particle_pos.shape)
    


