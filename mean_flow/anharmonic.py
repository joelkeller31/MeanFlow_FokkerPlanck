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
N = 10  # number of particles
d = 2   # dimension
A = 3 # interaction strength
r = 0.2  # particle size
D = 0.005  # diffusion coefficient
R = np.sqrt(5 *N)*r
B = D/R**2  # trap strength
D_sqrt = np.sqrt(D)
amp = lambda t: 2
freq = np.pi
drift = drifts.anharmonic_gaussian

Noisy = False 

circular = True

if bool(circular):
    compute_mut = lambda t: amp(t)*np.array([np.cos(freq*t), np.sin(freq*t)])
else:
    compute_mut = lambda t: amp(t)*np.array([np.cos(freq*t), 0])

## initial distribution parameters
force_args = (A, r, B, N, d, compute_mut)
mu0 = np.tile(compute_mut(0), N)
assert mu0.shape == (N*d,), f"Bad mu0 shape: {mu0.shape}"
sig0 = 0.01

experiment = "Anharmonic"


### more spread
n_hidden = 8
n_x_neurons=256
n_t_neurons=8 
n_epochs = 3500
learning_rate=1e-5
act = torch.nn.GELU

weight_decay=5e-6
batch_size =  1024

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
    trajs, ts = sim.generate_training_trajectories()
    sim.train_flow_matching(trajs)
    initial_conditions=torch.tensor(trajs[0, :, :-1]) 
    positions, indices = sim.mean_flow_trajectory_simulator(initial_conditions, 15, ts)
    sim.generate_validation_stats(trajectory=positions, true_traj_pts=trajs, indices=indices)
    sim.plot_trajectories_2d(trajectory=positions, true_traj_pts=trajs, n_x_neurons = n_x_neurons, n_t_neurons = n_t_neurons, n_hidden = n_hidden)
