from dataclasses import dataclass
from typing import Callable, Tuple, Union
import numpy as np
import torch 
from data import generate_training_data
from model import construct_mean_flow_model
from training import MeanFlowMatchingTrainer
State = np.ndarray
Time = float
from datetime import datetime 


import matplotlib.pyplot as plt

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MarginalFBTM():

    def __init__(self, data_dict: dict) -> None:
        self.__dict__ = data_dict.copy()
    
    sig0: float
    mu0: np.ndarray

    # system parameters
    drift: Callable[[State, Time], State]
    force_args: Tuple
    amp: Callable[[Time], float]
    freq: float
    dt: float
    D: np.ndarray
    D_sqrt: np.ndarray
    d: int
    N: int

    learning_rate: float

    rng: np.random.Generator

    n_time_steps: int
    batch_size: int

    def initialize_forcing(self) -> None:
        self.forcing = lambda x, t: self.drift(x, t, *self.force_args)


    def initialize_network_and_optimizer(self) -> None:
        
        self.network = construct_mean_flow_model(input_dim=self.d, output_dim=self.d, dim=self.n_x_neurons, n_hidden=self.n_hidden, time_embed_dim=self.n_t_neurons )
        self.opt = torch.optim.RAdam(
                    self.network.parameters(), 
                    self.learning_rate, 
                    weight_decay=1e-5
                )
        

    def generate_training_trajectories(self):
        x0 = self.mu0.reshape(self.N, self.d) + self.sig0 * self.rng.normal(size=(self.N, self.d))
        self.trajs = generate_training_data(self.n_time_steps, self.dt, self.N, self.d, 
                            self.D_sqrt, x0=x0, forcing=self.forcing, rng=self.rng, Noisy = self.Noisy)
        
        return self.trajs


    def plot_trajectories_2d(self, trajectory, true_traj_pts):
        
        trajectory = trajectory.cpu().detach().numpy() 

        tp, par, dim  = trajectory.shape
        plt.figure(figsize=(12, 10))
        
        for p in range(par):
            plt.plot(trajectory[:, p, 0], trajectory[:, p, 1], 
                    label=f'Particle {p+1} (Learned)', alpha=0.7)
            plt.scatter(true_traj_pts[:, p, 0], true_traj_pts[:, p, 1], label=f'Particle {p+1} (Euler Maruyama)', alpha=0.7)
        # # Mark start and end points
        for p in range(par):
            plt.scatter(trajectory[0, p, 0], trajectory[0, p, 1], 
                    marker='o', color='green', s=100, label='Start' if p == 0 else "")
            plt.scatter(trajectory[-1, p, 0], trajectory[-1, p, 1], 
                    marker='x', color='red', s=100, label='End' if p == 0 else "")
        title =  f'Generated Trajectories for {self.experiment} Experiment'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment}_N{par}_T{tp}_{timestamp}.png"
        plt.title(title)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/{filename}')

    def train_flow_matching(self, trajs):
        
        trainer = MeanFlowMatchingTrainer(model=self.network, opt = self.opt, train_trajs=trajs, batch_size=self.batch_size)
        trainer.train(num_epochs=self.n_epochs)

    

    def mean_flow_trajectory_simulator(self, initial_position, n_time_steps, t_start=0, t_end=1):
        model = self.network
        t_space = torch.linspace(t_start, t_end, n_time_steps, dtype=torch.float32)
        dt = (t_end - t_start) / n_time_steps

        n_particles, dim = initial_position.shape
        positions = torch.zeros(size=(n_time_steps, n_particles, dim))
        positions[0, :, :] = initial_position

        for n in range(n_time_steps - 1):
            current_t = t_space[n]
            current_pos = positions[n, :, :]
                        
            vel = model(
                x=current_pos, 
                t=current_t.expand(n_particles, 1), 
                r=(current_t - dt).expand(n_particles, 1)  
            )
            
            positions[n + 1, :, :] = positions[n, :, :] + dt * vel

        return t_space, positions