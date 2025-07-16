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
from matplotlib.lines import Line2D


import matplotlib.pyplot as plt

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def evaluate_analytical_cov(t): 
    return 100 * torch.exp(-2.3 * t) * (-0.025 * torch.exp(t) - 0.2 * torch.exp(1.3 * t) + 4.16334e-17 * torch.exp(2*t) + 0.475 * torch.exp(2.3 * t) + 6.935e-18 *torch.exp(2.667 * t))



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
        plt.ylim(-3, 2)
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
        plt.plot(non_zero_indices, non_zero_values, 'o-', markersize=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Entropy Production Rate')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment}_entropy_ts{timestamp}.png"
        plt.title('Entropy Production Rate During Training')
        plt.grid(True)
        plt.savefig(f'results/{filename}')
        plt.show()


    def plot_trajectories_2d(self, learned_traj, noisy_traj, clean_traj, n_x_neurons, n_t_neurons, n_hidden):
        """Plot three trajectory types side-by-side with fixed y-axis limits"""
        # Convert tensors to numpy if needed
        learned_traj = learned_traj.cpu().detach().numpy()
        noisy_traj = noisy_traj.cpu().detach().numpy() if isinstance(noisy_traj, torch.Tensor) else noisy_traj
        clean_traj = clean_traj.cpu().detach().numpy() if isinstance(clean_traj, torch.Tensor) else clean_traj

        os.makedirs('results', exist_ok=True)
        tp, par, dim = learned_traj.shape
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        
        # Style parameters
        colors = plt.cm.tab10(np.linspace(0, 1, par))
        marker_size = 60
        line_width = 1.5
        alpha = 0.8
        
        # # Set consistent y-axis limits for all plots
        # if self.experiment == "Harmonic": 
        #     y_limits = (-3.25, 2.0)
        # if self.experiment == "Anharmonic": 
        #     y_limits = (-1, 4.0)

        # Plot 1: Noisy Trajectories (Input Data)
        for p in range(par):
            ax1.plot(noisy_traj[:, p, 0], noisy_traj[:, p, 1],
                    color=colors[p], linestyle='-', linewidth=line_width, alpha=alpha)
            ax1.scatter(noisy_traj[0, p, 0], noisy_traj[0, p, 1],
                    color=colors[p], marker='o', s=marker_size, 
                    edgecolor='black', zorder=3)
            ax1.scatter(noisy_traj[-1, p, 0], noisy_traj[-1, p, 1],
                    color=colors[p], marker='X', s=marker_size,
                    edgecolor='black', zorder=3)
        ax1.set_title('Noisy Trajectories (Input Data)', fontsize=14)
        ax1.set_xlabel("X Position", fontsize=12)
        ax1.set_ylabel("Y Position", fontsize=12)
        # ax1.set_ylim(y_limits)  # Set y-axis limits
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Learned Trajectories (Model Output)
        for p in range(par):
            ax2.plot(learned_traj[:, p, 0], learned_traj[:, p, 1],
                    color=colors[p], linestyle='--', linewidth=line_width, alpha=alpha)
            ax2.scatter(learned_traj[0, p, 0], learned_traj[0, p, 1],
                    color=colors[p], marker='o', s=marker_size,
                    edgecolor='black', zorder=3)
            ax2.scatter(learned_traj[-1, p, 0], learned_traj[-1, p, 1],
                    color=colors[p], marker='X', s=marker_size,
                    edgecolor='black', zorder=3)
        ax2.set_title('Learned Trajectories (Model Output)', fontsize=14)
        ax2.set_xlabel("X Position", fontsize=12)
        # ax2.set_ylim(y_limits)  # Set y-axis limits
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Noise-Free Trajectories (Ground Truth)
        for p in range(par):
            ax3.plot(clean_traj[:, p, 0], clean_traj[:, p, 1],
                    color=colors[p], linestyle=':', linewidth=line_width, alpha=alpha)
            ax3.scatter(clean_traj[0, p, 0], clean_traj[0, p, 1],
                    color=colors[p], marker='o', s=marker_size,
                    edgecolor='black', zorder=3)
            ax3.scatter(clean_traj[-1, p, 0], clean_traj[-1, p, 1],
                    color=colors[p], marker='X', s=marker_size,
                    edgecolor='black', zorder=3)
        ax3.set_title('Noise-Free Trajectories (Ground Truth)', fontsize=14)
        ax3.set_xlabel("X Position", fontsize=12)
        # ax3.set_ylim(y_limits)  # Set y-axis limits
        ax3.grid(True, alpha=0.3)
        
        # Create unified legend
        legend_elements = []
        for p in range(par):
            legend_elements.append(Line2D([0], [0], 
                                    color=colors[p], 
                                    lw=2, 
                                    label=f'Particle {p+1}'))
        
        # Add start/end markers to legend
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', label='Start',
                markerfacecolor='black', markersize=10),
            Line2D([0], [0], marker='X', color='w', label='End',
                markerfacecolor='black', markersize=10)
        ])
        
        # Add line style indicators
        legend_elements.extend([
            Line2D([0], [0], color='k', linestyle='-', lw=2, label='Noisy'),
            Line2D([0], [0], color='k', linestyle='--', lw=2, label='Learned'),
            Line2D([0], [0], color='k', linestyle=':', lw=2, label='Noise-Free')
        ])
        
        fig.legend(handles=legend_elements, 
                loc='center right', 
                bbox_to_anchor=(1.15, 0.5),
                fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment}_trajectory_comparison_N{n_hidden}_X{n_x_neurons}_T{n_t_neurons}_{timestamp}.png"
        plt.savefig(f'results/{filename}', dpi=300, bbox_inches='tight')
        plt.close()

    def compare_noisy_and_clean(self, noisy_traj, clean_traj):
        os.makedirs('results', exist_ok=True)
        tp, par, dim = noisy_traj.shape
        
        plt.figure(figsize=(14, 12))
        
        colors = plt.cm.tab10(np.linspace(0, 1, par))
        
        for p in range(par):
            plt.plot(noisy_traj[:, p, 0], noisy_traj[:, p, 1], 
                    color=colors[p], linestyle='-', 
                    linewidth=1.5, alpha=0.5,
                    label=f'Particle {p+1} (Noisy)')
            
            plt.plot(clean_traj[:, p, 0], clean_traj[:, p, 1], 
                    color=colors[p], linestyle='--', 
                    linewidth=2.5, alpha=0.9,
                    label=f'Particle {p+1} (Clean)')
            
            plt.scatter(noisy_traj[0, p, 0], noisy_traj[0, p, 1], 
                    color=colors[p], marker='o', s=100, 
                    edgecolor='black', zorder=3)
            plt.scatter(clean_traj[-1, p, 0], clean_traj[-1, p, 1], 
                    color=colors[p], marker='X', s=100, 
                    edgecolor='black', zorder=3)

        plt.scatter([], [], marker='o', color='black', s=100, 
                label='Start', edgecolor='black')
        plt.scatter([], [], marker='X', color='black', s=100, 
                label='End', edgecolor='black')
        
        title = f'Trajectory Comparison: {self.experiment}'
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel("X Position", fontsize=12)
        plt.ylabel("Y Position", fontsize=12)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                fontsize=10, framealpha=1)
        
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='datalim')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/{self.experiment}_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_means_and_covs(self, covs_trace, covs_eig, analytical_cov, timepoints):
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        covs_trace = [item.cpu().detach().numpy() if isinstance(item, torch.Tensor) else item 
                        for item in covs_trace]
        analytical_cov = [item.cpu().detach().numpy() if isinstance(item, torch.Tensor) else item 
                        for item in analytical_cov]
        timepoints = [item.cpu().detach().numpy() if isinstance(item, torch.Tensor) else item 
                        for item in timepoints]
        
        plt.figure(figsize=(12, 5))  
        
        # Plot Trace Covariance vs Analytical
        plt.plot(timepoints, covs_trace, 'o-', markersize=4, label='Numerical', alpha=0.7)
        plt.plot(timepoints, analytical_cov, 'x-', markersize=4, label='Analytical', alpha=0.7)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Covariance (Trace)', fontsize=12)
        plt.title('Covariance Comparison: Numerical vs Analytical Solution', fontsize=14)
        plt.ylim(0, 100)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"results/{self.experiment}_covs_ts{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
            
    def train_flow_matching(self, clean_trajs, noisy_trajs):
        "learns the deterministic vector field"
        trainer = MeanFlowMatchingTrainer(model=self.network, opt = self.opt, clean_trajs=clean_trajs, noisy_trajs=noisy_trajs, batch_size=self.batch_size, n_epochs=self.n_epochs)
        print('Training Clean Vector Field')
        self.training_entropy = trainer.train()
        return self.network

    def mean_flow_trajectory_simulator(self, initial_position, n_time_steps, ts, model):
        t_space = torch.linspace(0, 1, n_time_steps).to(Device)
        dt = 1 / (n_time_steps - 1)  
        
        n_particles, dim = initial_position.shape
        positions = torch.zeros(size=(n_time_steps, n_particles, dim)).to(Device)
        positions[0, :, :] = initial_position
        
        covs_datapoint_1 = torch.zeros(size=(n_time_steps, 1))
        covs_datapoint_2 = torch.zeros(size=(n_time_steps, 1))
        analytical_cov = torch.zeros(size=(n_time_steps, 1))

        current_pos = positions[0, :, :]
        eig = torch.linalg.eigvalsh(torch.cov(current_pos))
        if self.experiment == "Harmonic":
            covs_datapoint_1[0] = eig[-1]
            covs_datapoint_2[0] = torch.trace(torch.cov(current_pos))
            analytical_cov[0] = evaluate_analytical_cov(t_space[0])
        else:
            covs_datapoint_1[0] = torch.cov(current_pos)[0, 0]
            covs_datapoint_2[0] = torch.cov(current_pos)[1, 1]
            analytical_cov[0] = evaluate_analytical_cov(t_space[0])

        for n in range(n_time_steps - 1):
            current_t = t_space[n+1]  
            
            vel = model(
                x=positions[n, :, :], 
                t=t_space[n].expand(n_particles, 1), 
                r=(t_space[n] - dt).expand(n_particles, 1)  
            )
            
            positions[n + 1, :, :] = positions[n, :, :] + dt * vel
            
            current_pos = positions[n+1, :, :]
            eig = torch.linalg.eigvalsh(torch.cov(current_pos))
            if self.experiment == "Harmonic":
                covs_datapoint_1[n+1] = eig[-1]
                covs_datapoint_2[n+1] = torch.trace(torch.cov(current_pos))
                analytical_cov[n+1] = evaluate_analytical_cov(current_t)
            else:
                covs_datapoint_1[n+1] = torch.cov(current_pos)[0, 0]
                covs_datapoint_2[n+1] = torch.cov(current_pos)[1, 1]
                analytical_cov[n+1] = evaluate_analytical_cov(current_t)

        return positions, covs_datapoint_1, covs_datapoint_2, analytical_cov, t_space
