"""
Nicholas M. Boffi
7/29/22

Drift terms for score-based transport modeling.
"""

from typing import Callable, Tuple
import torch 
import numpy as np

compute_particle_diffs = torch.vmap(
        torch.vmap(lambda x, y: x - y, in_dims=(0, None), out_dims=0),
        in_dims=(None, 0), out_dims=1
    )

def active_swimmer(
    xv: np.ndarray,
    t: float,
    gamma: float
) -> np.ndarray:
    """Active swimmer example."""
    del t
    x, v = xv
    return np.array([-x**3 + v, -gamma*v])


def harmonic_trap(
    x: np.ndarray,
    t: float,
    compute_mut: Callable[[float], np.ndarray],
    N: int,
    d: int
) -> np.ndarray:
    """Forcing for particles in a harmonic trap with harmonic repulsion."""
    mut = compute_mut(t)
    particle_pos = x.reshape((N, d))
    particle_forces = -0.5*(particle_pos + mut[None, :] \
            + np.mean(particle_pos, axis=0)[None, :])

    return particle_forces.ravel()


def gaussian_interaction(
    xs: np.ndarray,
    A: float,
    r: float,
    N: int
) -> np.ndarray:
    particle_diffs = xs[:, None, :] - xs[None, :, :]  # (N, N, d)
    dist_sq = np.sum(particle_diffs**2, axis=2)
    gauss_facs = np.exp(-dist_sq / (2*r**2))
    interaction = (A/(N*r**2)) * np.sum(particle_diffs * gauss_facs[:, :, None], axis=1)
    return interaction


def anharmonic_gaussian(
    x: np.ndarray,
    t: float,
    A: float,
    r: float,
    B: float,
    N: int,
    d: int,
    compute_mut: Callable[[float], np.ndarray]
) -> np.ndarray:
    """Corrected implementation matching the SDE"""
    particle_pos = x.reshape((N, d))
    beta_t = compute_mut(t)
    diff = particle_pos - beta_t[None, :]
    diff_norms_sq = np.sum(diff**2, axis=1)
    
    # Trap force (note 4B factor)
    trap_force = -4 * B * diff * diff_norms_sq[:, None]
    
    # Interaction force
    interaction = gaussian_interaction(particle_pos, A, r, N)
    
    return (trap_force + interaction).ravel()

def anharmonic(
    x: np.ndarray,
    t: float,
    compute_mut: Callable[[float], np.ndarray]
) -> np.ndarray:
    """Single particle in an anharmonic trap."""
    diff = x - compute_mut(t)
    return -diff * (diff @ diff)


def anharmonic_harmonic(
    x: np.ndarray,
    t: float,
    A: float,
    B: float,
    N: int,
    d: int,
    compute_mut: Callable[[float], np.ndarray],
) -> np.ndarray:
    """Harmonically-interacting particles in an anharmonic trap."""
    particle_pos = x.reshape((N, d))
    diff = particle_pos - compute_mut(t)[None, :]
    diff_norms = np.sum(diff**2, axis=1)
    xbar = np.mean(particle_pos, axis=0)
    return np.ravel(-B*diff*diff_norms[:, None] + A*(particle_pos - xbar[None, :]))