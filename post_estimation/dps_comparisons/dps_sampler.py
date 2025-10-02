#%%
import sys
import os
import __main__ as main

# set the cwd to be the repo root
if os.getcwd().endswith('post_estimation'):
    os.chdir('..')
elif os.getcwd().endswith('dps_comparisons'):
    os.chdir('../..')


device = 'cuda'
import numpy as np 
from tqdm import tqdm 
import toml
import configargparse
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


#use inline plotting
#matplotlib.use('TkAgg')
#import the gamma function
from scipy.special import gamma

log_wandb = True
n_var_samples = 20

from post_estimation.dps_comparisons.model import ScoreNet
from post_estimation.dps_comparisons.vp_sde import marginal_prob_std, marginal_prop_mean, diffusion_coeff, drift_coeff

beta_min = 0.1
beta_max = 20 
noise_sd = torch.tensor([1.],device='cuda')

marginal_prob_std_fn = functools.partial(marginal_prob_std, 
                                        beta_min=beta_min, beta_max=beta_max)
marginal_prob_mean_fn = functools.partial(marginal_prop_mean, 
                                        beta_min=beta_min, beta_max=beta_max)
diffusion_coeff_fn = functools.partial(diffusion_coeff, 
                                        beta_min=beta_min, beta_max=beta_max)
drift_coeff_fn = functools.partial(drift_coeff, 
                                        beta_min=beta_min, beta_max=beta_max)

score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
ckpt = torch.load('post_estimation/dps_comparisons/checkpoints/mri_score_model_1e-05_170.pth', map_location=device)
score_model.load_state_dict(ckpt)
score_model.eval()
score_model.to(device)

def middle_inds(im_size,n_pts):
    x = torch.arange(im_size**2).reshape(im_size,im_size)
    n_pts_from_middle = torch.sqrt(torch.tensor(n_pts / 4)).int().item()
    x_midpoint = int(im_size/2)
    A = x[x_midpoint-n_pts_from_middle:x_midpoint+n_pts_from_middle,x_midpoint-n_pts_from_middle:x_midpoint+n_pts_from_middle].flatten()
    return A

def forward_op(indices, x):
    x = x[:, 0] + x[:, 1] * 1j
    x = torch.fft.fft2(x)
    x = torch.fft.fftshift(x)
    x = torch.stack([x.real, x.imag], dim=-1)
    x = x.view(x.shape[0], -1, 2)
    # Apply partial sampling by selecting a subset of the Fourier coefficients
    mask = torch.zeros_like(x)
    mask[:, indices] = 1
    x = x * mask
    x = x.reshape(128,128,2).permute(2,0,1).unsqueeze(0)
    #x = x[:, indices]
    return x

def lik_score(indices, x,x_dat):
    x = x[:, 0] + x[:, 1] * 1j
    x = torch.fft.fft2(x)
    x = torch.fft.fftshift(x)
    x = torch.stack([x.real, x.imag], dim=-1)
    x = x.view(x.shape[0], -1, 2)
    # Apply partial sampling by selecting a subset of the Fourier coefficients
    mask = torch.zeros_like(x)
    mask[:, indices] = 1
    x = x * mask
    x_dat = x_dat.squeeze(0).permute(1,2,0).reshape(128**2,2).unsqueeze(0)
    x = x - x_dat
    x = x.reshape(128,128,2).permute(2,0,1)
    x = x[0] + x[1] * 1j
    x = torch.fft.fftshift(x)
    x = torch.fft.ifft2(x)
    x = torch.stack([x.real, x.imag], dim=-1)
    return x.permute(2,0,1).unsqueeze(0)


def parse_config(default_config_path: str) -> dict:
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to config file')
    args, _ = parser.parse_known_args()

    if args.config:
        # Config file provided via command line
        config_path = args.config
        config_dict = toml.load(config_path)
    else:
        config_dict = toml.load(default_config_path)

    return config_dict
        

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
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



opt = parse_config(default_config_path = 'post_estimation/recon_configs/config6')

#log_wandb = opt['wandb']['log_wandb']
project_name = opt['wandb']['project_name']
dataset = opt['data']['dataset'] 
image_size = opt['data']['image_size']
num_channels = opt['data']['num_channels']
device = opt['general']['device']
batch_size = opt['training']['batch_size']
latent_dimension_size = opt['model']['latent_dimension_size']
local_connection_dist = opt['model']['local_connection_dist']
use_attention = opt['model']['use_attention']
use_group_norm = opt['model']['use_group_norm']
init_decoder_var = torch.tensor(opt['model']['init_decoder_var'])
optimiser = opt['training']['optimiser']
n_data_points = opt['data']['n_data_points']
load_trained_supn_model = opt['model']['load_trained_supn_model']
supn_model_load_path = opt['model']['supn_model_load_path']
train_schedule = opt['train_schedule']
cov_pix = (opt['cov_pix_ind']['row'],opt['cov_pix_ind']['col'])
measurement_prop = opt['recon_pars']['measurement_prop']
num_var_samples = opt['recon_pars']['num_var_samples']
test_set = opt['recon_pars']['test_set']
im_ind = opt['recon_pars']['im_ind']
#noise_sd = opt['recon_pars']['noise_sd']
mask_type = opt['recon_pars']['mask_type']

#noise_sd = torch.tensor(noise_sd,device=device)
train_data = np.load(f'fastMRI/complex_train_images/knee_fastMRI_train_128_cleaned_complex.npy').transpose(0,2,3,1).astype(np.float32)
test_data = np.load(f'fastMRI/complex_test_images/knee_fastMRI_test_128_cleaned_complex.npy').transpose(0,2,3,1).astype(np.float32)

train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)

#set seed for train_loader
im_ind = 5
x_true = train_dataset[im_ind][0].to(device).unsqueeze(0)

torch.manual_seed(0)

for measurement_prop in [0.4,0.025,0.05,0.1,0.2]:

    #
    A = middle_inds(128,measurement_prop*128**2).int()

    M = torch.zeros(128**2).to('cuda')
    M[A] = 1
    plt.imshow(M.cpu().detach().numpy().reshape(128,128), cmap='gray')

    x_dat = forward_op(A,x_true) 
    noise = noise_sd*torch.randn_like(M)*M
    x_dat = x_dat + (noise*M).reshape(128,128)

    def alpha(t):
        return 1 - marginal_prob_std_fn(t)**2

    batch_size = 1
    T = 1.
    eps = 1e-5
    num_steps = 8000
    t = torch.ones(batch_size, device=device)
    time_steps = torch.linspace(T, eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    # data consistency strength parameter
    tunable = step_size*(num_steps/100)
    
    n_samples = 20

    samples = []
    time_step = time_steps[0]

    # dps sampling loop
    for j in range(n_samples):
        init_x = torch.randn(batch_size, 2, 128, 128, device=device) * T 
        x = init_x
        for time_step in tqdm(time_steps):      
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            s = score_model(x, batch_time_step)
            alph = alpha(batch_time_step)
            x_est = (1/torch.sqrt(alph))*(x + (1-alph)*s)
            grad_log_lik = lik_score(A,x_est,x_dat)
            g = diffusion_coeff(batch_time_step)
            f = drift_coeff(batch_time_step)[:, None, None, None]
            mean_x = x - (f*x - (g**2)[:, None, None, None] * (score_model(x, batch_time_step))) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)   
            x = x - grad_log_lik*tunable
            x = x.detach()   

        samples.append(x.cpu().detach().numpy())

    samples = np.array(samples)

    np.save(f'../samples/diff_samples_{measurement_prop}.npy',samples)

#%%


#%%

# %%
