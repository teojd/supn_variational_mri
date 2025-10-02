#%% Imports and set up functions

import sys
import os
import __main__ as main

if os.getcwd().endswith('post_estimation'):
    os.chdir('..')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(main.__file__))+'/supn_base')


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

from supn_base.supn_distribution import SUPN
from models.supn_models.colour import VAE
from supn_base.supn_data import SUPNData, get_num_off_diag_weights, get_num_cross_channel_weights
from utils.smooth_clip import softclip

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
import re


#use inline plotting
matplotlib.use('Agg')
#import the gamma function
from scipy.special import gamma

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
    x = x[:, indices]
    return x


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


#%% Load config for the run that is being visualised
opt = parse_config(default_config_path = 'post_estimation/recon_configs/config6')
experiment_name = ''

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
noise_sd = opt['recon_pars']['noise_sd']
mask_type = opt['recon_pars']['mask_type']

test_data = np.load(f'fastMRI/complex_test_images/knee_fastMRI_test_128_cleaned_complex.npy').transpose(0,2,3,1).astype(np.float32)
train_data = np.load(f'fastMRI/complex_train_images/knee_fastMRI_train_128_cleaned_complex.npy').transpose(0,2,3,1).astype(np.float32)


#%% Get library of settings and their corresponding experiment names
def build_experiment_lib_from_folder(folder_path):
    """
    Scans the given folder for subfolders matching the pattern
    'middle_{mask_prop}_{im_ind}_map_run_{date}_{time}' and returns a dictionary
    mapping 'middle_{mask_prop}_{im_ind}' to '{date}_{time}'.
    """
    experiment_lib = {}
    pattern = re.compile(r'middle_([0-9.]+)_([0-9]+)_map_run_([0-9]+_[0-9]+)')
    for name in os.listdir(folder_path):
        match = pattern.match(name)
        if match:
            mask_prop, im_ind, timestring = match.groups()
            key = f'middle_{mask_prop}_{im_ind}'
            experiment_lib[key] = timestring
    return experiment_lib

experiment_lib = build_experiment_lib_from_folder('post_estimation/checkpoints/separable_variational')

mask_type = 'middle'
mask_prop = 0.05
im_ind = 10

im_ind2 = 812

if im_ind == 5:
    dataset = CustomDataset(train_data)
else:
    dataset = CustomDataset(test_data)

x_true = dataset[im_ind][0].to(device).unsqueeze(0)
x_true2 = dataset[im_ind2][0].to(device).unsqueeze(0)
settings_str =  f'{mask_type}_{mask_prop}_{im_ind}'
experiment_name = experiment_lib[settings_str]
filename = f'{settings_str}_map_run_{experiment_name}'
qz_pars = torch.load(f'post_estimation/checkpoints/separable_variational/middle_{mask_prop}_{im_ind}_map_run_{experiment_name}/60000/qz_pars.pth')
supn_model = torch.load(f'post_estimation/checkpoints/separable_variational/middle_{mask_prop}_{im_ind}_map_run_{experiment_name}/60000/supn_model.pth')

#%% Load prior and sample from it
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Create sample data

supn_model = VAE(image_size = image_size,
                latent_dim = latent_dimension_size,
                local_connection_dist = local_connection_dist,
                num_channels = num_channels,
                use_attention = use_attention,
                use_group_norm = use_group_norm,
                init_decoder_var = init_decoder_var,
                ).to(device)
model_path = 'supn_train/checkpoints/pretrained.pth'

supn_model.load_state_dict(torch.load(model_path))

prior_dist = supn_model.decode(qz_pars[0].T)
pri_mean = prior_dist.mean
pri_sample = prior_dist.sample(1)
pri_resid = pri_sample - prior_dist.mean


p2 = torch.zeros(1,1,2,128,128).cuda()

#%%

coord = [[58,39]]
#coord = [[80,93]]
p2 = torch.zeros(1,1,2,128,128).cuda()
clamp = [-0.1,0.2]
#clamp = [-0.03,0.03]

for row_n, col_n in coord:
    rhs = torch.zeros(2*128**2).reshape(1,1,2,128,128)
    rhs[0,0,0,row_n,col_n] = 10
    rhs = rhs.to(device)

    p1 = prior_dist.chol_cov_mult(rhs, solve_with_upper=True)
    p2 = p2 + prior_dist.chol_cov_mult(p1, solve_with_upper=False)
    p2[0,0,0,row_n,col_n] = 0

pri_corr = p2




#%%
supn_model = torch.load(f'post_estimation/checkpoints/separable_variational/middle_0.025_22_map_run_20250223_135259/60000/supn_model.pth')

q_supn_data = SUPNData(mean=supn_model[0],
                       log_diag=supn_model[1],
                       off_diag=supn_model[2],
                       cross_ch=supn_model[3],
                       local_connection_dist=4)

q_supn_dist = SUPN(supn_data=q_supn_data)



post_samples = []
post_means = []
post_corrs = []
#im_ind = 22
for prop in [0.05, 0.2, 0.4]:
        settings_str =  f'{mask_type}_{prop}_{im_ind}'
        experiment_name = experiment_lib[settings_str]
        folder_name = f'{settings_str}_map_run_{experiment_name}'
        supn_model = torch.load(f'post_estimation/checkpoints/separable_variational/{folder_name}/60000/supn_model.pth')

        q_supn_data = SUPNData(mean=supn_model[0],
                            log_diag=supn_model[1],
                            off_diag=supn_model[2],
                            cross_ch=supn_model[3],
                            local_connection_dist=4)

        q_supn_dist.supn_data = q_supn_data
        post_samples.append(q_supn_dist.sample(1))
        post_means.append(q_supn_dist.mean)


        p2 = torch.zeros(1,1,2,128,128).cuda()

        coord = [[58,39]]

        p2 = torch.zeros(1,1,2,128,128).cuda()
        clamp = [-0.02,0.02]
        for row_n, col_n in coord:
            rhs = torch.zeros(2*128**2).reshape(1,1,2,128,128)
            rhs[0,0,0,row_n,col_n] = 10
            rhs = rhs.to(device)

            p1 = q_supn_dist.chol_cov_mult(rhs, solve_with_upper=True)
            p2 = p2 + q_supn_dist.chol_cov_mult(p1, solve_with_upper=False)
            p2[0,0,0,row_n,col_n] = 0
        post_corrs.append(p2)

mask_type = 'middle'
mask_prop = 0.025
im_ind = im_ind2

for prop in [0.05, 0.2, 0.4]:
        settings_str =  f'{mask_type}_{prop}_{im_ind}'
        experiment_name = experiment_lib[settings_str]
        folder_name = f'{settings_str}_map_run_{experiment_name}'
        supn_model = torch.load(f'post_estimation/checkpoints/separable_variational/{folder_name}/60000/supn_model.pth')

        q_supn_data = SUPNData(mean=supn_model[0],
                            log_diag=supn_model[1],
                            off_diag=supn_model[2],
                            cross_ch=supn_model[3],
                            local_connection_dist=4)

        q_supn_dist.supn_data = q_supn_data
        post_samples.append(q_supn_dist.sample(1))
        post_means.append(q_supn_dist.mean)


        p2 = torch.zeros(1,1,2,128,128).cuda()

        coord = [[58,39]]

        p2 = torch.zeros(1,1,2,128,128).cuda()
        clamp = [-0.02,0.02]
        for row_n, col_n in coord:
            rhs = torch.zeros(2*128**2).reshape(1,1,2,128,128)
            rhs[0,0,0,row_n,col_n] = 10
            rhs = rhs.to(device)

            p1 = q_supn_dist.chol_cov_mult(rhs, solve_with_upper=True)
            p2 = p2 + q_supn_dist.chol_cov_mult(p1, solve_with_upper=False)
            p2[0,0,0,row_n,col_n] = 0
        post_corrs.append(p2)

#%%

coord = [[58,39]]

p2 = torch.zeros(1,1,2,128,128).cuda()
clamp = [-0.1,0.2]
for row_n, col_n in coord:
    rhs = torch.zeros(2*128**2).reshape(1,1,2,128,128)
    rhs[0,0,0,row_n,col_n] = 10
    rhs = rhs.to(device)

    p1 = prior_dist.chol_cov_mult(rhs, solve_with_upper=True)
    p2 = p2 + prior_dist.chol_cov_mult(p1, solve_with_upper=False)
    p2[0,0,0,row_n,col_n] = 0








#%%
inversion_recons = []
fourier_space_noises = []
masks = []
for measure_prop in [0.025, 0.05, 0.1, 0.2, 0.4]:
    A = middle_inds(128,measure_prop*128**2).int()
    M = torch.zeros(128**2)
    M[A] = 1
    #plt.imshow(M.cpu().detach().numpy().reshape(128,128), cmap='gray')



    x_complex = x_true[:, 0] + x_true[:, 1] * 1j
    x = torch.fft.fft2(x_complex)
    x = torch.fft.fftshift(x)
    x = torch.stack([x.real, x.imag], dim=-1)
    x = x.view(x.shape[0], -1, 2)
    noise  = noise_sd * torch.randn_like(x)
    x = x + noise
    # Apply partial sampling by selecting a subset of the Fourier coefficients
    mask = torch.zeros_like(x)
    mask[:, A] = 1
    x = x * mask
    four_noise = mask*noise


    M = torch.zeros(128**2)
    M[A] = 1

    x = x.reshape(128,128,2).permute(2,0,1)
    four_noise = four_noise.reshape(128,128,2).permute(2,0,1)
    #plt.imshow(x[1].cpu().detach().numpy().reshape(128,128), cmap='gray')


    x = x[0] + x[1] * 1j
    x = torch.fft.fftshift(x)
    x = torch.fft.ifft2(x)
    x = torch.stack([x.real, x.imag], dim=-1)
    four_noise = four_noise[0] + four_noise[1] * 1j
    four_noise = torch.fft.fftshift(four_noise)
    four_noise = torch.fft.ifft2(four_noise)
    four_noise = torch.stack([four_noise.real, four_noise.imag], dim=-1)

    inversion_recons.append(x)
    fourier_space_noises.append(four_noise)
    masks.append(M.reshape([128,128]))


#%% Load prior and sample from it
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Create sample data

supn_model = VAE(image_size = image_size,
                latent_dim = latent_dimension_size,
                local_connection_dist = local_connection_dist,
                num_channels = num_channels,
                use_attention = use_attention,
                use_group_norm = use_group_norm,
                init_decoder_var = init_decoder_var,
                ).to(device)
model_path = 'supn_train/checkpoints/pretrained.pth'

supn_model.load_state_dict(torch.load(model_path))

prior_dist = supn_model.decode(qz_pars[0].T)
pri_mean = prior_dist.mean
pri_sample = prior_dist.sample(1)
pri_resid = pri_sample - prior_dist.mean


p2 = torch.zeros(1,1,2,128,128).cuda()

#%%

coord = [[58,39]]

p2 = torch.zeros(1,1,2,128,128).cuda()
clamp = [-0.1,0.2]
for row_n, col_n in coord:
    rhs = torch.zeros(2*128**2).reshape(1,1,2,128,128)
    rhs[0,0,0,row_n,col_n] = 10
    rhs = rhs.to(device)

    p1 = prior_dist.chol_cov_mult(rhs, solve_with_upper=True)
    p2 = p2 + prior_dist.chol_cov_mult(p1, solve_with_upper=False)
    p2[0,0,0,row_n,col_n] = 0


#%%
# make two figures with 5 subplots each one for the residuals post_samples - post_means and one for
# the samples
clamp0 = [-0.06,0.06]
clamp = [-1.0,1.0]
clamp_1 = [-0.15,0.15]
norm0 = mcolors.TwoSlopeNorm(vmin=clamp0[0], vcenter=0, vmax=clamp0[1])
norm = mcolors.TwoSlopeNorm(vmin=clamp[0], vcenter=0, vmax=clamp[1])
norm_1 = mcolors.TwoSlopeNorm(vmin=clamp_1[0], vcenter=0, vmax=clamp_1[1])

def norm_func(x,scale=1.0):
    max_val = x.abs().max()
    clamp = [-max_val, max_val]
    return mcolors.TwoSlopeNorm(vmin=clamp[0]*scale, vcenter=0, vmax=clamp[1]*scale)

fig, axs = plt.subplots(3, 6, figsize=(4.15*5, 10.5))
axs[0,1-1].imshow(pri_resid[0,0,0].detach().cpu().clamp(-1,1), cmap='gray')
axs[0,3-1].imshow(pri_mean[0,0].detach().cpu().clamp(-1,1), cmap='gray')
axs[0,4-1].imshow(pri_sample[0,0,0].detach().cpu().clamp(-1,1), cmap='gray')
axs[0,5-1].imshow(x_true[0,0].detach().cpu().clamp(-1,1), cmap='gray')
axs[0,2-1].imshow(pri_corr[0,0,0].detach().cpu().clamp(-0.2,0.2), cmap='gray',norm=norm0)
axs[0, 0].axis('off')
axs[0, 1].axis('off')
axs[0, 2].axis('off')
axs[0, 3].axis('off')
axs[0, 4].axis('off')

for i in range(3):
    #axs[i+1, 0+1].imshow(masks[i].cpu().detach().numpy().reshape(128,128), cmap='gray')
    #axs[i+1, 0+1].axis('off')
    axs[i, 3-1].imshow((post_samples[i][0,0,0].clamp(-1,1) - post_means[i][0,0]).detach().cpu(), cmap='gray', norm = norm_func((post_samples[i][0,0,0] - post_means[i][0,0]).detach().cpu(),0.85) )
    axs[i, 3-1].axis('off')
    axs[i, 1-1].imshow(post_samples[i][0,0,0].detach().cpu().clamp(-1,1), cmap='gray')
    axs[i, 1-1].axis('off')
    axs[i, 4-1].imshow(post_samples[i][0,0,0].detach().cpu().clamp(-1,1), cmap='gray')
    axs[i, 4-1].axis('off')
    axs[i, 5-1].imshow(inversion_recons[i][:,:,0].real.detach().cpu().clamp(-1,1), cmap='gray')
    axs[i, 5-1].axis('off')
    axs[i, 2-1].imshow(x_true[0,0].cpu() - post_samples[i][0,0,0].real.detach().cpu().clamp(-1,1), cmap='gray',norm=norm_func(x_true[0,0].cpu() - post_samples[i][0,0,0].real.detach().cpu()))
    axs[i, 2-1].axis('off')
    axs[i, 5].axis('off')


for i in range(3):
    #axs[i+1, 0+1].imshow(masks[i].cpu().detach().numpy().reshape(128,128), cmap='gray')
    #axs[i+1, 0+1].axis('off')
    axs[i, 3+2].imshow((post_samples[i+3][0,0,0].clamp(-1,1) - post_means[i+3][0,0]).detach().cpu(), cmap='gray', norm = norm_func((post_samples[i+3][0,0,0].clamp(-1,1) - post_means[i+3][0,0]).detach().cpu(),0.85) )
    axs[i, 3+2].axis('off')
    axs[i, 1+2].imshow(post_samples[i+3][0,0,0].detach().cpu().clamp(-1,1), cmap='gray')
    axs[i, 1+2].axis('off')
    axs[i, 2+2].imshow(x_true2[0,0].cpu() - post_samples[i+3][0,0,0].real.detach().cpu().clamp(-1,1), cmap='gray',norm=norm_func(x_true2[0,0].cpu() - post_samples[i+3][0,0,0].real.detach().cpu().clamp(-1,1)))
    axs[i, 2+2].axis('off')
    axs[i, 5].axis('off')

fig.tight_layout()


axs[0,0].set_title('Reconstruction', fontsize=20)
axs[0,1].set_title('True residual', fontsize=20)
axs[0,2].set_title('Model residual', fontsize=20)
axs[0,3].set_title('Reconstruction', fontsize=20)
axs[0,4].set_title('True residual', fontsize=20)
axs[0,5].set_title('Model residual', fontsize=20)
#axs[0,3].set_title('Prior sample', fontsize=20)
#axs[0,4].set_title('True image', fontsize=20)

#axs[1,0].set_title('Posterior Residual', fontsize=20)
#axs[1,1].set_title('Posterior pixelwise correlation', fontsize=20)
#axs[1,2].set_title('Posterior mean', fontsize=20)
#axs[1,3].set_title('Posterior sample', fontsize=20)
#axs[1,4].set_title('Zero-filled reconstruction', fontsize=20)

# add row labels
axs[0,0].text(-0.1, 0.5, "5%", transform=axs[0,0].transAxes, 
             fontsize=20, verticalalignment='center', rotation=90)
axs[1,0].text(-0.1, 0.5, "20%", transform=axs[1,0].transAxes, 
             fontsize=20, verticalalignment='center', rotation=90)
axs[2,0].text(-0.1, 0.5, "40%", transform=axs[2,0].transAxes, 
             fontsize=20, verticalalignment='center', rotation=90)
#axs[3,0].text(-0.1, 0.5, "10%", transform=axs[3,0].transAxes, 
#             fontsize=20, verticalalignment='center', rotation=90)
#axs[4,0].text(-0.1, 0.5, "20%", transform=axs[4,0].transAxes, 
#             fontsize=20, verticalalignment='center', rotation=90)
#axs[5,0].text(-0.1, 0.5, "40%", transform=axs[5,0].transAxes, 
#             fontsize=20, verticalalignment='center', rotation=90)


#axs[0, 0].set_ylabel(f'5%')
#axs[1, 0].set_ylabel(f'20%')
#axs[2, 0].set_ylabel(f'40%')
#axs[4, 0].set_ylabel(f'20%')
#axs[5, 0].set_ylabel(f'40%')

# add a red star to axs[0,1] at coords (80,40) and (60,30)
#axs[0,1].scatter(39,58,marker='*',color='red',s=200)

fig.tight_layout()

plt.subplots_adjust(wspace=0.05, hspace=0.05)

#mak this directory if it does not exist
if not os.path.exists('figures'):
    os.makedirs('figures')
plt.savefig(f'figures/residual_comp.pdf')
