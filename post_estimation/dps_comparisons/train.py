#%%
import numpy as np 
from tqdm import tqdm 

import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision import datasets
from vp_sde import marginal_prob_std, marginal_prop_mean, diffusion_coeff, drift_coeff

#%%
import numpy as np
import torch
from pathlib import Path
import torch
import sys
import os
import __main__ as main
if os.path.basename(os.getcwd()) == 'examples':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.abspath(main.__file__))+'/../supn_base')
else:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.abspath(main.__file__))+'/supn_base')

sys.path.append('/home/teo/Documents/supn_project')
sys.path.append('/home/teo/Documents/supn_project/examples/diff_train')

import numpy as np 
from tqdm import tqdm 
np.float_ = np.float64
np.complex_ = np.complex128

import torch
torch.cuda.is_available()
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision import datasets

import wandb
from time import sleep
from datetime import datetime
import matplotlib
import time as time
import random 
from math import ceil, factorial
from datetime import datetime
import matplotlib
from matplotlib import pyplot as plt

log_wandb = False

if log_wandb:
    wandb.init(project='mri_langevin_score')
    # log all hyperparameters and save code
    wandb.run.log_code(".")

#%%

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data  # Assume this is a list of NumPy arrays
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5), (0.5,0.5))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = self.transform(x)  # Convert to tensor
        y = torch.tensor(0, dtype=torch.long)  # Dummy label 0 as tensor
        return x, y


training_data = np.load(f'/home/teo/Documents/supn_project/datasets/fastMRI/complex_train_images/knee_fastMRI_train_128_cleaned_complex.npy').transpose(0,2,3,1).astype(np.float32)

#%%
model_type = 'vector'

if model_type=='potential':
    from model import PotentialScoreNet as ScoreNet
    model_path = 'potential_net1_1e-05.pth'
elif model_type=='vector':
    from model import ScoreNet
    model_path = 'vector_transformer_net1_0_1e-05.pth'



device = "cuda"

def loss_fn(model, x, marginal_prob_std, marginal_prop_mean, eps=1e-5, T=1.):

    """The loss function for training score-based generative models.

    Args:
        model: A PyTorch model instance that represents a 
        time-dependent score-based model.
        x: A mini-batch of training data.    
        marginal_prob_std: A function that gives the standard deviation of 
        the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (T - eps) + eps  
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
 
    mean = marginal_prop_mean(random_t)
    perturbed_x = mean[:, None, None, None] * x + z * std[:, None, None, None]
    #score = model(perturbed_x, random_t)
    #loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    score = score_model.forward_unscaled(perturbed_x, random_t)  
    loss = torch.mean(2*(1-random_t)*torch.sum((score + z)**2, dim=(1,2,3)))

    return loss


beta_min = 0.1
beta_max = 20 

marginal_prob_std_fn = functools.partial(marginal_prob_std, 
                                        beta_min=beta_min, beta_max=beta_max)
marginal_prob_mean_fn = functools.partial(marginal_prop_mean, 
                                        beta_min=beta_min, beta_max=beta_max)
diffusion_coeff_fn = functools.partial(diffusion_coeff, 
                                        beta_min=beta_min, beta_max=beta_max)
drift_coeff_fn = functools.partial(drift_coeff, 
                                        beta_min=beta_min, beta_max=beta_max)

score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
score_model = score_model.to(device)



print("Number of Parameters: ", sum([p.numel() for p in score_model.parameters()]))


# load pre-trained score model
ckpt = torch.load('/home/teo/Documents/MRI_diff/MNIST/checkpoints/mri_score_model_40.pth', map_location=device)
score_model.load_state_dict(ckpt)

n_epochs = 1000
batch_size = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
train_dataset = CustomDataset(training_data)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
a = next(iter(data_loader))[0]

# if './checkpoints' does not exist, create it
if not os.path.exists('./checkpoints2'):
    os.makedirs('./checkpoints2')


def Euler_Maruyama_sampler(score_model, 
                           drift_coeff,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=500, 
                           device=device, 
                           eps=1e-5,
                           T = 1.):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns:
    Samples.    
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 2, 128, 128, device=device) * T 
  time_steps = torch.linspace(T, eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  for time_step in tqdm(time_steps):      
    batch_time_step = torch.ones(batch_size, device=device) * time_step

    g = diffusion_coeff(batch_time_step)
    f = drift_coeff(batch_time_step)[:, None, None, None]
    mean_x = x - (f*x - (g**2)[:, None, None, None] * score_model(x, batch_time_step)) * step_size
    x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)   
    x = x.detach()   

  # Do not include any noise in the last sampling step.
  return mean_x

def sample_and_plot():
    samples = Euler_Maruyama_sampler(score_model, 
                           drift_coeff=drift_coeff_fn,
                           diffusion_coeff=diffusion_coeff_fn, 
                           batch_size=sample_batch_size, 
                           num_steps=400, 
                           device=device, 
                           eps=1e-5,
                           T=T)

    ## Sample visualization.
    samples = samples.clamp(-1.0, 1.0)
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    sample_grid = make_grid(samples, nrow=4)

    plt.imshow(torch.concat([sample_grid.permute(1, 2, 0).cpu()[:,:,0],sample_grid.permute(1, 2, 0).cpu()[:,:,1]],axis=0),cmap = 'gray')
    plt.axis('off')
    plt.savefig('sample.png')
    fig = plt.gcf()
    wandb.log({'Batch_sample_test': wandb.Image(fig)})


#%%

sample_batch_size = 4
T = 1.
## Generate samples using the specified sampler.

accumulation_steps = 4
for lr, epochs, batch_size in [[1e-4,50,6],[5e-5,50,6],[2.5e-5,50,6],[1e-5,500,6]]:    
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    optimizer = Adam(score_model.parameters(), lr=lr)
    print("Start Training")
    for epoch in range(epochs+1):
        avg_loss = 0.
        num_items = 0
        progress_bar = tqdm(data_loader, desc=f'Train Epoch: {epoch}')
        for batch_idx, (x, _) in enumerate(progress_bar):
            x = x.to(device)    
            loss = loss_fn(score_model, x[:6], marginal_prob_std=marginal_prob_std_fn,marginal_prop_mean=marginal_prob_mean_fn, T=1.)
            loss = loss / accumulation_steps
            loss.backward()    
            avg_loss += loss.item() * x.shape[0]
            if epoch % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
            num_items += x.shape[0]
            progress_bar.set_postfix({'loss': loss.item()})
        # Print the averaged training loss so far.
        print('Average Loss at epoch {}: {:5f}'.format(epoch, avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        wandb.log({'loss': avg_loss / num_items})
        if epoch % 2 == 0:
            sample_and_plot()
        if epoch % 10 == 0:
            torch.save(score_model.state_dict(), f'./checkpoints2/mri_score_model_{lr}_{int(epoch)}.pth')
#%%



