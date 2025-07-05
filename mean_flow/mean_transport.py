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
import os 

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
        
        self.network = construct_mean_flow_model(input_dim=self.d, output_dim=self.d, dim=self.n_x_neurons, n_hidden=self.n_hidden, act= self.act, time_embed_dim=self.n_t_neurons)
        self.opt = torch.optim.RAdam(
                    self.network.parameters(), 
                    self.learning_rate, 
                    weight_decay=1e-5
                )
        

    def generate_training_trajectories(self):
        x0 = self.mu0.reshape(self.N, self.d) + self.sig0 * self.rng.normal(size=(self.N, self.d))
        self.trajs, ts = generate_training_data(self.n_time_steps, self.dt, self.N, self.d, 
                            self.D_sqrt, x0=x0, forcing=self.forcing, rng=self.rng, Noisy = self.Noisy)
        
        return self.trajs, ts

        
    def generate_validation_stats(self, trajectory, indices, true_traj_pts): 
        true_traj_pts = torch.tensor(true_traj_pts[indices]).to(Device)
        
        for p in range(trajectory.size(1)): 
            pred = trajectory[:, p, :]
            true = true_traj_pts[:, p, :-1]
            
            # Basic statistics
            mean_error = torch.mean(pred - true)
            std_error = torch.std(pred - true)
            
            # Distance-based metrics
            mse = torch.mean((pred - true) ** 2)
            rmse = torch.sqrt(mse)
            mae = torch.mean(torch.abs(pred - true))
                        
            # Euclidean distance per time step
            euclidean_distances = torch.norm(pred - true, dim=1)
            mean_euclidean_distance = torch.mean(euclidean_distances)
            max_euclidean_distance = torch.max(euclidean_distances)
            
            # Relative error (avoid division by zero)
            true_norms = torch.norm(true, dim=1)
            relative_errors = euclidean_distances / torch.clamp(true_norms, min=1e-8)
            mean_relative_error = torch.mean(relative_errors)
            
            # Correlation coefficient
            pred_flat = pred.flatten()
            true_flat = true.flatten()
            correlation = torch.corrcoef(torch.stack([pred_flat, true_flat]))[0, 1]
            
            print(f'Particle {p} Statistics:')
            print(f'  Mean Error: {mean_error:.6f}')
            print(f'  Std Error: {std_error:.6f}')
            print(f'  MSE: {mse:.6f}')
            print(f'  RMSE: {rmse:.6f}')
            print(f'  MAE: {mae:.6f}')
            print(f'  Mean Euclidean Distance: {mean_euclidean_distance:.6f}')
            print(f'  Max Euclidean Distance: {max_euclidean_distance:.6f}')
            print(f'  Mean Relative Error: {mean_relative_error:.6f}')
            print(f'  Correlation: {correlation:.6f}')

    def plot_trajectories_2d(self, trajectory, true_traj_pts):
        trajectory = trajectory.cpu().detach().numpy() 

        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)

        tp, par, dim = trajectory.shape
        plt.figure(figsize=(12, 10))
        
        for p in range(par):
            plt.plot(trajectory[:, p, 0], trajectory[:, p, 1], 
                    label=f'Particle {p+1} (Learned)', alpha=0.7)
            plt.scatter(true_traj_pts[:, p, 0], true_traj_pts[:, p, 1], 
                    label=f'Particle {p+1} (Euler Maruyama)', alpha=0.7)

        # Mark start and end points
        for p in range(par):
            plt.scatter(trajectory[0, p, 0], trajectory[0, p, 1], 
                    marker='o', color='green', s=100, label='Start' if p == 0 else "")
            plt.scatter(trajectory[-1, p, 0], trajectory[-1, p, 1], 
                    marker='x', color='red', s=100, label='End' if p == 0 else "")
        
        title = f'Generated Trajectories for {self.experiment} Experiment'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment}_N{par}_T{tp}_{timestamp}.png"
        plt.title(title)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/{filename}')
        plt.close()

    def train_flow_matching(self, trajs):
        
        trainer = MeanFlowMatchingTrainer(model=self.network, opt = self.opt, train_trajs=trajs, batch_size=self.batch_size, n_epochs=self.n_epochs)
        trainer.train()

    

    def mean_flow_trajectory_simulator(self, initial_position, n_time_steps, ts):
        model = self.network
        indices = [i * int(ts.size/n_time_steps) for i in range( n_time_steps)  ]
        t_space = torch.tensor(ts[indices], dtype=torch.float32).to(Device)
        print(f't space shape: {t_space.shape}')
        dt = 1 / n_time_steps

        n_particles, dim = initial_position.shape
        positions = torch.zeros(size=(n_time_steps, n_particles, dim)).to(Device)
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

        return positions, indices