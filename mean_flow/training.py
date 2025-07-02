import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Optional
from torch.nn.utils import clip_grad_norm_




class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass


    def train(self, num_epochs: int, device: torch.device, **kwargs) -> torch.Tensor:
        # Start
        self.model.to(device)
        opt = self.opt
        self.model.train()


        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(epoch, **kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')

        # Finish
        self.model.eval()



def time_check(t_times, r_times): 
    tp = t_times.shape
    print('checking times ... ')
    print(f'tp shape: {tp}')
    for t in range(tp): 
        assert t_times[t] > r_times[t]
        
class MeanFlowMatchingTrainer(Trainer):
    def __init__(self, 
                 opt,
                 model: torch.nn.Module, 
                 train_trajs: list,
                 batch_size: int,
                ):
        
        super().__init__(model)
        self.train_trajs = train_trajs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt = opt

    def sample_t_and_r_indices(self, size, large_interval_prob=0.3):  
        # for anharmonic: 0.3
        # for harmonic: 0.15


        bins = torch.linspace(0, 5000, steps=size+1)


        t = torch.zeros(size)
        r = torch.zeros(size)
        for i in range(size):
            lower = bins[i]
            upper = bins[i+1]
            
            if torch.rand(1) < large_interval_prob and i < size-1:  # Large interval case
                max_jump = min(4, size-i-1)
                bin_jump = torch.randint(1, max_jump+1, (1,)).item() if max_jump > 0 else 1
                jump_lower = bins[i + bin_jump]
                jump_upper = bins[i + bin_jump + 1]
                
                t[i] = torch.rand(1) * (upper - lower) + lower
                r[i] = torch.rand(1) * (jump_upper - jump_lower) + jump_lower
            else:  # Small interval case
                samples = torch.rand(2) * (upper - lower) + lower
                t[i], r[i] = samples.max(), samples.min()
        
        # Ensure t > r by swapping where needed
        swap_mask = t < r
        t[swap_mask], r[swap_mask] = r[swap_mask], t[swap_mask]
        
        return t.unsqueeze(1), r.unsqueeze(1)
    
    def get_space_and_time_for_idx(self, t_indices, r_indices, trajs): 
        train_trajs_tensor = torch.tensor(trajs)  # Convert to tensor
        
        # Extract positions and times
        x_t = train_trajs_tensor[t_indices.long(), :, :2].float()  # Shape: (batch_size, 2)
        x_r = train_trajs_tensor[r_indices.long(), :, :2].float()  # Shape: (batch_size, 2)
        tau_t = train_trajs_tensor[t_indices.long(), :, 2:3].float()  # shape ([128, 1, 15, 1])
        tau_r = train_trajs_tensor[r_indices.long(), :, 2:3].float()  
        return x_t, tau_t, x_r, tau_r
    
    def get_train_loss(self, epoch: int, **kwargs) -> torch.Tensor:
        t_idx, r_idx = self.sample_t_and_r_indices(self.batch_size)
        train_trajs_tensor = torch.tensor(self.train_trajs)  
        
        
        x_t, tau_t, x_r, tau_r = self.get_space_and_time_for_idx(t_idx, r_idx, self.train_trajs)

        
        ep = torch.rand_like(tau_t)  

        t_var = tau_r + ep * (tau_t - tau_r)  
        
        # normalize times in (0,1)
        alpha = (t_var - tau_r) / (tau_t - tau_r + 1e-8)


        # interpolated_samples = x_r * (1 - alpha) + x_t * alpha
        

        # what if instead, we made an interpolation of the points we sampled, and then interpolated_samples just evaluates it at that time

        interpolated_samples = x_r * (1 - alpha)**2 + x_t * alpha**2 + 2 * (x_r + x_t) * alpha * (1 - alpha)
        
        velocity = (x_t - x_r) / (tau_t - tau_r + 1e-8)

        # Compute model output and its time derivative
        def model_forward(x, t):
            return self.model.forward(x, t, tau_r)  # Assuming model takes (x, t, r)


        vectors = (torch.zeros_like(interpolated_samples), torch.ones_like(t_var), torch.zeros_like(tau_r))

        # Compute the function value and its jvp
        u, dudt = torch.func.jvp(
            lambda x, t, r: model_forward(x, t),  
            (interpolated_samples, t_var, tau_r),  
            vectors,  # Vectors for jvp
        )

        u_tgt = velocity + (tau_t - tau_r) * dudt.detach()  

        return torch.nn.functional.mse_loss(u, u_tgt)


    def train(self, num_epochs: int, **kwargs):


        clip_max_norm= 3

        self.model.to(self.device)
        opt = self.opt
        
        with tqdm(range(num_epochs), desc="Training") as pbar:
            for epoch in pbar:
                opt.zero_grad()
                loss = self.get_train_loss(epoch, **kwargs)
                loss.backward()
                
                if clip_max_norm is not None:
                    total_norm = clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=clip_max_norm,
                        norm_type=2.0  # L2 normb
                    )
                    pbar.set_postfix(loss=loss.item(), grad_norm=total_norm.item())
                else:
                    pbar.set_postfix(loss=loss.item())
                opt.step()

        self.model.eval()


