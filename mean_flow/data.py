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
    clean_trajs = np.zeros((n_steps + 1, N, d + 1)) 
    noisy_trajs = np.zeros((n_steps + 1, N, d + 1))  
    
    clean_trajs[0, :, :d] = x0
    noisy_trajs[0, :, :d] = x0  
    
    # Set initial times correctly
    clean_trajs[0, :, d] = ts[0]
    noisy_trajs[0, :, d] = ts[0]
    
    for i in range(n_steps):
        current_positions = clean_trajs[i, :, :d]
        noisy_positions = noisy_trajs[i, :, :d]

        clean_step, noisy_step = euler_maruyama_step(
            clean_particle_pos=current_positions.copy(),
            noisy_particle_pos=noisy_positions.copy(),  
            t=ts[i],
            D_sqrt=D_sqrt,
            dt=dt,
            rng=rng,
            noisy=Noisy,
            forcing=forcing
        )
        
        # Update trajectories
        clean_trajs[i+1, :, :d] = clean_step
        noisy_trajs[i+1, :, :d] = noisy_step
        clean_trajs[i+1, :, d] = ts[i+1]
        noisy_trajs[i+1, :, d] = ts[i+1]

    return clean_trajs, noisy_trajs, ts

def euler_maruyama_step(
    clean_particle_pos: np.ndarray,  
    noisy_particle_pos: np.ndarray,  
    t: float,
    D_sqrt: float,
    dt: float,
    forcing: Callable[[np.ndarray, float], np.ndarray],
    rng: np.random.Generator,
    noisy: bool = False, 
) -> np.ndarray:

    clean_step = clean_particle_pos + dt * forcing(clean_particle_pos, t).reshape(clean_particle_pos.shape)
    noisy_step= noisy_particle_pos + dt * forcing(noisy_particle_pos, t).reshape(noisy_particle_pos.shape) + np.sqrt(2 * dt) * D_sqrt * rng.normal(size=noisy_particle_pos.shape)
    return clean_step, noisy_step



