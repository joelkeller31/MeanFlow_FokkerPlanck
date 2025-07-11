from dataclasses import dataclass
from typing import Callable, Tuple, Union
import numpy as np
import torch 
from data import generate_training_data
from model import construct_mean_flow_model, construct_noisy_model
from training import MeanFlowMatchingTrainer, NoisyResidualTrainer
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
    weight_decay: float
    learning_rate: float

    rng: np.random.Generator

    n_time_steps: int
    batch_size: int

    def initialize_forcing(self) -> None:
        self.forcing = lambda x, t: self.drift(x, t, *self.force_args)


    def initialize_network_and_optimizer(self) -> None:
        
        self.network = construct_mean_flow_model(input_dim=self.d, output_dim=self.d, dim=self.n_x_neurons, n_hidden=self.n_hidden, act= self.act, time_embed_dim=self.n_t_neurons)
        self.opt = torch.optim.AdamW(
                    self.network.parameters(), 
                    self.learning_rate, 
                    weight_decay=self.weight_decay
                )
        

    def generate_training_trajectories(self):
        x0 = self.mu0.reshape(self.N, self.d) + self.sig0 * self.rng.normal(size=(self.N, self.d))
        self.clean_trajs, self.noisy_trajs, ts = generate_training_data(self.n_time_steps, self.dt, self.N, self.d, 
                            self.D_sqrt, x0=x0, forcing=self.forcing, rng=self.rng, Noisy = self.Noisy)
        
        return self.clean_trajs, self.noisy_trajs, ts

        
    def generate_validation_stats(self, trajectory, indices, true_traj_pts): 
        true_traj_pts = torch.tensor(true_traj_pts[indices]).to(Device)
        

        #iterate through the particles
        for p in range(trajectory.size(1)): 
            pred = trajectory[:, p, :]
            true = true_traj_pts[:, p, :-1]
            
            mean_error = torch.mean(pred - true)
            std_error = torch.std(pred - true)
            
            mse = torch.mean((pred - true) ** 2)
            
                        
            # Euclidean distance per time step
            euclidean_distances = torch.norm(pred - true, dim=1)
            mean_euclidean_distance = torch.mean(euclidean_distances)
            max_euclidean_distance = torch.max(euclidean_distances)
            
            
            
            print(f'Particle {p+1} Statistics:')
            # print(f'  Mean Error: {mean_error:.6f}')
            # print(f'  Std Error: {std_error:.6f}')
            print(f'  MSE: {mse:.6f}')
            
            print(f'  Mean Euclidean Distance: {mean_euclidean_distance:.6f}')
            print(f'  Max Euclidean Distance: {max_euclidean_distance:.6f}')
            

    def plot_entropy(self):

        os.makedirs('results', exist_ok=True)
        plt.figure(figsize=(12, 10))
        
        # Convert tensors to numpy arrays for plotting
        if isinstance(self.training_entropy[0], torch.Tensor):
            # If tensors are on GPU, move to CPU first, then convert to numpy
            entropy_values = [tensor.cpu().detach().numpy() for tensor in self.training_entropy]
        else:
            # If already numpy arrays or scalars
            entropy_values = self.training_entropy
        
        # Simple line plot
        plt.plot(entropy_values)
        plt.xlabel('Epoch/Iteration')
        plt.ylabel('Entropy')
        plt.title('Training Entropy Over Time')
        plt.grid(True)
        plt.savefig('results/entropy_plot.png')

    # def plot_entropy(self):

    #     os.makedirs('results', exist_ok=True)
    #     plt.figure(figsize=(12, 10))
        
    #     # Convert tensors to numpy arrays for plotting
    #     if isinstance(self.training_entropy[0], torch.Tensor):
    #         # If tensors are on GPU, move to CPU first, then convert to numpy        
    #         entropy_values = [item.cpu().detach().numpy() if isinstance(item, torch.Tensor) else item 
    #              for item in self.training_entropy]
    #     else:
    #         # If already numpy arrays or scalars
    #         entropy_values = self.training_entropy
        
    #     # Simple line plot
    #     plt.plot(entropy_values)
    #     plt.xlabel('Epoch/Iteration')
    #     plt.ylabel('Entropy')
    #     plt.title('Training Entropy Over Time')
    #     plt.grid(True)
    #     plt.savefig('results/entropy_plot.png')

    def plot_entropy(self):
        os.makedirs('results', exist_ok=True)
        plt.figure(figsize=(12, 10))
        # plt.ylim(-1, 2)
        # Convert tensors to numpy arrays for plotting
        entropy_values = [item.cpu().detach().numpy() if isinstance(item, torch.Tensor) else item 
                        for item in self.training_entropy]
        
        # Filter out zeros and create corresponding epoch indices
        non_zero_indices = []
        non_zero_values = []
        
        for i, val in enumerate(entropy_values):
            if val != 0:  # Only include non-zero entropy values
                non_zero_indices.append(i)
                non_zero_values.append(val)
        
        if not non_zero_values:
            print("No non-zero entropy values to plot")
            return
        
        # Plot with actual epoch numbers on x-axis
        plt.plot(non_zero_indices, non_zero_values, 'o-', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Entropy')
        plt.title('Training Entropy Over Time (Non-zero values only)')
        plt.grid(True)
        plt.savefig('results/entropy_plot.png')
        plt.show()
    def plot_trajectories_2d(self, trajectory, true_traj_pts, n_x_neurons, n_t_neurons, n_hidden):
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
        filename = f"{self.experiment}_N{n_hidden}_X{n_x_neurons}_T_{n_t_neurons}_ts{timestamp}.png"
        plt.title(title)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/{filename}')
        plt.close()



    def compare_noisy_and_clean(self, noisy_traj, clean_traj):
        os.makedirs('results', exist_ok=True)
        tp, par, dim = noisy_traj.shape
        
        # Create figure with larger size
        plt.figure(figsize=(14, 12))
        
        # Use a colormap for better distinction between particles
        colors = plt.cm.tab10(np.linspace(0, 1, par))
        
        for p in range(par):
            # Plot noisy trajectory (with lower alpha)
            plt.plot(noisy_traj[:, p, 0], noisy_traj[:, p, 1], 
                    color=colors[p], linestyle='-', 
                    linewidth=1.5, alpha=0.5,
                    label=f'Particle {p+1} (Noisy)')
            
            # Plot clean trajectory (with solid line)
            plt.plot(clean_traj[:, p, 0], clean_traj[:, p, 1], 
                    color=colors[p], linestyle='--', 
                    linewidth=2.5, alpha=0.9,
                    label=f'Particle {p+1} (Clean)')
            
            # Mark start and end points
            plt.scatter(noisy_traj[0, p, 0], noisy_traj[0, p, 1], 
                    color=colors[p], marker='o', s=100, 
                    edgecolor='black', zorder=3)
            plt.scatter(clean_traj[-1, p, 0], clean_traj[-1, p, 1], 
                    color=colors[p], marker='X', s=100, 
                    edgecolor='black', zorder=3)

        # Only add start/end labels once
        plt.scatter([], [], marker='o', color='black', s=100, 
                label='Start', edgecolor='black')
        plt.scatter([], [], marker='X', color='black', s=100, 
                label='End', edgecolor='black')
        
        # Add title and labels
        title = f'Trajectory Comparison: {self.experiment}'
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel("X Position", fontsize=12)
        plt.ylabel("Y Position", fontsize=12)
        
        # Improve legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                fontsize=10, framealpha=1)
        
        # Add grid and equal aspect ratio
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='datalim')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/{self.experiment}_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()



    def train_flow_matching(self, clean_trajs):
        "learns the deterministic vector field"
        trainer = MeanFlowMatchingTrainer(model=self.network, opt = self.opt, train_trajs=clean_trajs, batch_size=self.batch_size, n_epochs=self.n_epochs)
        print('Training Clean Vector Field')
        trainer.train()
        return self.network


    def train_residual(self, noisy_trajs):

        trainer = NoisyResidualTrainer(model=self.noisy_network, opt = self.opt, train_trajs=noisy_trajs, clean_trajs= self.clean_trajs,batch_size=self.batch_size, n_epochs=self.n_epochs)
        print('Training Noisy Residual')
        self.training_entropy = trainer.train()

        return self.noisy_network

    def mean_flow_trajectory_simulator(self, initial_position, n_time_steps, ts):
        model = self.noisy_network
        indices = [i * int(ts.size/n_time_steps) for i in range( n_time_steps)  ]
        t_space = torch.tensor(ts[indices], dtype=torch.float32).to(Device)
        print(f' t space')
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
    


    def initialize_noisy_model(self, learned_vector_field) -> None:
        self.noisy_network = construct_noisy_model(input_dim=self.d, output_dim=self.d, dim=self.n_x_neurons, n_hidden=self.n_hidden, act= self.act, time_embed_dim=self.n_t_neurons, clean_vector_field=learned_vector_field)
        