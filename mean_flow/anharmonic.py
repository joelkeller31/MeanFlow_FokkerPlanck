from dataclasses import dataclass
from typing import Callable, Tuple, Union
import numpy as np
import torch 
# import optax
import drifts as drifts 
from mean_transport import MarginalFBTM
import argparse
import matplotlib.pyplot as plt 


Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d = 2
D = 0.25
dt = 1e-3
tf = 5
n_time_steps = int(tf / dt)
store_fac = 25


## configure random seed
repeatable_seed = True

if repeatable_seed:
    rng = np.random.default_rng(42)
else:
    rng = np.random.default_rng()

#work on this
# N = 10  # number of particles
# d = 2   # dimension
# A = 1.5 # interaction strength
# r = 0.2  # particle size
# D = .75 # diffusion coefficient
# R = np.sqrt(5 *N)*r
# B = D/R**2  # trap strength


N = 10       # Keep same particle count
d = 2        # Keep 2D system
A = 10      # Increased from 1.5 (stronger attractive interactions)
r = 0.15     # Reduced from 0.2 (smaller particle size = denser packing)
D = 0.2      # Reduced from 0.75 (less diffusion/dispersion)
R = np.sqrt(5*N)*r  # Automatically smaller due to smaller r
B = 0.8*D/R**2 


D_sqrt = np.sqrt(D)
amp = lambda t: 3
freq = np.pi
drift = drifts.anharmonic_gaussian

Noisy = True 

circular = True

if bool(circular):
    compute_mut = lambda t: amp(t)*np.array([np.cos(freq*t), np.sin(freq*t)])
else:
    compute_mut = lambda t: amp(t)*np.array([np.cos(freq*t), 0])

## initial distribution parameters
force_args = (A, r, B, N, d, compute_mut)
mu0 = np.tile(compute_mut(0), N)
assert mu0.shape == (N*d,), f"Bad mu0 shape: {mu0.shape}"
sig0 = 0.15

experiment = "Anharmonic"





n_hidden = 4                
n_x_neurons = 128             
n_t_neurons = 32            
n_epochs = 800         
learning_rate = 3e-5         
act = torch.nn.GELU          
weight_decay = 1e-5
batch_size = 128              



def construct_simulation():

    sim_params = {
    "sig0": sig0,
    "mu0": mu0,
    "drift": drift,
    "force_args": force_args,
    "amp": amp,
    "freq": freq,
    "dt": dt,
    "D": D,
    "D_sqrt": np.sqrt(D),
    "N": N,
    "d": d,
    "Noisy": Noisy, 
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "n_hidden": n_hidden,
    "n_x_neurons": n_x_neurons,
    "n_t_neurons": n_t_neurons,
    "n_epochs": n_epochs,
    "act": act,
    "rng": rng,
    "n_time_steps": n_time_steps,
    "experiment": experiment,
    "batch_size": batch_size, 
    }
    
    tot_params = {** sim_params}
    sim = MarginalFBTM(tot_params)

    return sim


if __name__ == '__main__':
    # start = time.time()
    sim = construct_simulation()
    sim.initialize_network_and_optimizer()
    sim.initialize_forcing()
    print('Generating Trajectories: ...')
    clean_trajs, noisy_trajs, ts = sim.generate_training_trajectories()
    # sim.compare_noisy_and_clean(clean_trajs, noisy_trajs)
    learned_vector_field = sim.train_flow_matching(clean_trajs)
    sim.initialize_noisy_model(learned_vector_field)
    sim.train_residual(noisy_trajs)
    sim.plot_entropy()
    # initial_conditions=torch.tensor(noisy_trajs[0, :, :-1]) 
    # positions, indices = sim.mean_flow_trajectory_simulator(initial_conditions, 15, ts)
    # sim.generate_validation_stats(trajectory=positions, true_traj_pts=noisy_trajs, indices=indices)
    # sim.plot_trajectories_2d(trajectory=positions, true_traj_pts=noisy_trajs, n_x_neurons = n_x_neurons, n_t_neurons = n_t_neurons, n_hidden = n_hidden)
