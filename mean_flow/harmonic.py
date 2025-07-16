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

######## Configuation Parameters #########
## physical, non-forcing parameters
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


## harmonic system
N = 15
mask = np.ones(N*d)
amp = lambda t: 2
freq = 1
compute_mut = lambda t: np.array([amp(t)*np.cos(np.pi*freq*t), amp(t)*np.sin(np.pi*freq*t)])
drift = drifts.harmonic_trap
force_args = (compute_mut, N, d)
mu0 = np.tile(compute_mut(0), N)
Noisy = True

experiment = "Harmonic"

sig0 = 0.25


n_hidden =   2        
n_x_neurons = 64             
n_t_neurons = 8           
n_epochs = 2500  

learning_rate = 3e-4
act = torch.nn.GELU          
weight_decay = 6e-6
batch_size = 512     

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
    "experiment": experiment,
    "n_time_steps": n_time_steps,
    "batch_size": batch_size 
    }
    
    tot_params = {** sim_params}
    sim = MarginalFBTM(tot_params)

    return sim




if __name__ == '__main__':
    # start = time.time()
    sim = construct_simulation()
    sim.initialize_network_and_optimizer()
    sim.initialize_forcing()
    clean_trajs, noisy_trajs, ts = sim.generate_training_trajectories()
    # sim.compare_noisy_and_clean(clean_trajs, noisy_trajs)
    learned_vector_field = sim.train_flow_matching(clean_trajs, noisy_trajs)
    initial_conditions=torch.tensor(noisy_trajs[0, :, :-1]) 

    positions, means, covs, analytical_cov, timepoints = sim.mean_flow_trajectory_simulator(initial_conditions, 5, ts, model = learned_vector_field)
    sim.plot_trajectories_2d(learned_traj=positions, noisy_traj=noisy_trajs, clean_traj=clean_trajs, n_x_neurons = n_x_neurons, n_t_neurons = n_t_neurons, n_hidden = n_hidden)
    sim.plot_means_and_covs(means, covs, analytical_cov, timepoints)
    # sim.plot_entropy()
