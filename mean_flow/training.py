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


        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(epoch, **kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')

        self.model.eval()




        
class MeanFlowMatchingTrainer(Trainer):
    def __init__(self, 
                 opt,
                 model: torch.nn.Module, 
                 train_trajs: list,
                 batch_size: int,
                 n_epochs: int
                ):
        
        super().__init__(model)
        self.train_trajs = train_trajs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt = opt
        self.n_epochs = n_epochs


    def sample_t_and_r_indices(self, batch_size, epoch):
        # learn trajectories in a 750-index wide window 
        t = torch.randint(low=750, high=5000, size=(batch_size,))
        r = t - 750 
        return t.unsqueeze(1), r.unsqueeze(1)


    def get_space_and_time_for_idx(self, t_indices, r_indices, trajs): 
        train_trajs_tensor = torch.tensor(trajs)  
        
        # Extract positions and times
        x_t = train_trajs_tensor[t_indices.long(), :, :2].squeeze(1).float().to(self.device)  
        x_r = train_trajs_tensor[r_indices.long(), :, :2].squeeze(1).float().to(self.device)  
        tau_t = train_trajs_tensor[t_indices.long(), :, 2:3].squeeze(1).float().to(self.device)  
        tau_r = train_trajs_tensor[r_indices.long(), :, 2:3].squeeze(1).float().to(self.device)  
       

        return x_t, tau_t, x_r, tau_r
    
    def get_train_loss(self, epoch: int, **kwargs) -> torch.Tensor:

        t_idx, r_idx = self.sample_t_and_r_indices(self.batch_size, epoch)
        train_trajs_tensor = torch.tensor(self.train_trajs)  
        
        
        x_t, tau_t, x_r, tau_r = self.get_space_and_time_for_idx(t_idx, r_idx, self.train_trajs)


        ep = torch.rand_like(tau_t)  

        t_var = tau_r + ep * (tau_t - tau_r)  
        
        # normalize times in (0,1) - helps interpolation step 
        alpha = (t_var - tau_r) / (tau_t - tau_r + 1e-8)

        interpolated_samples = x_r * (1 - alpha) + x_t * alpha
        
        
        velocity = (x_t - x_r) / (tau_t - tau_r + 1e-8)

        def model_forward(x, t):
            return self.model.forward(x, t, tau_r) 


        vectors = (torch.zeros_like(interpolated_samples), torch.ones_like(t_var), torch.zeros_like(tau_r))

        # Compute the function value and its jvp
        u, dudt = torch.func.jvp(
            lambda x, t, r: model_forward(x, t),  
            (interpolated_samples, t_var, tau_r),  
            vectors,  # differentiate wrt time 
        )

        u_tgt = velocity + (tau_t-tau_r) * dudt.detach()  

        mse_loss = torch.nn.functional.mse_loss(u, u_tgt)

        return mse_loss 

    def train(self, **kwargs):


        clip_max_norm= 0.1

        self.model.to(self.device)
        opt = self.opt
        
        with tqdm(range(self.n_epochs), desc="Training") as pbar:
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


