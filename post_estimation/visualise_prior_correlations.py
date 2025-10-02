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

log_wandb = True

def corner_inds(im_size,n_pts):
    x = torch.arange(im_size**2).reshape(im_size,im_size)    
    n_pts_per_corner = torch.tensor(n_pts / 4)
    sq_size = int(torch.sqrt(n_pts_per_corner))
    A_UL = x[:sq_size,:sq_size]
    A_UR = x[:sq_size,-sq_size:]
    A_LL = x[-sq_size:,:sq_size]
    A_LR = x[-sq_size:,-sq_size:]
    A = torch.cat((A_UL,A_UR,A_LL,A_LR),dim=1).flatten()
    return A

def middle_inds(im_size,n_pts):
    x = torch.arange(im_size**2).reshape(im_size,im_size)
    n_pts_from_middle = torch.sqrt(torch.tensor(n_pts / 4)).int().item()
    x_midpoint = int(im_size/2)
    A = x[x_midpoint-n_pts_from_middle:x_midpoint+n_pts_from_middle,x_midpoint-n_pts_from_middle:x_midpoint+n_pts_from_middle].flatten()
    return A

def horiz_inds(im_size,n_pts):
    x = torch.arange(im_size**2).reshape(im_size,im_size)
    n_pts_from_middle = torch.sqrt(torch.tensor(n_pts / 4)).int().item()
    x_midpoint = int(im_size/2)
    A = x[x_midpoint-n_pts_from_middle:x_midpoint+n_pts_from_middle,:].flatten()
    return A


def forward_op(A,x):
    x = x.reshape([2,x.shape[1]**2]).T
    return A@x

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

def log_prior_z(z):
    return -0.5*torch.sum(z**2,axis=1)

def log_prior_sig_eps(sig_eps):
    #alpha = 0.1
    #beta = 5
    #B = gamma(alpha)*gamma(beta)/gamma(alpha+beta)
    #return torch.sum((alpha-1)*torch.log(sig_eps) + (beta-1)*torch.log(1-sig_eps) - np.log(B))
    return -torch.sum(0.5*sig_eps**2)

def log_prior_x_given_z(x,z,supn_model,batch = False):
    if x.shape[0] > 1:
        batch = True
    supn_dist = supn_model.decode(z.T)
    #recon_logvar = softclip(-recon_logvar)
    LogProb = supn_dist.log_prob(x)
    #log_prob_from_sparse_chol_prec(x=x,
    #                                    mean=recon_x,
    #                                    log_diag_weights=recon_logvar,
    #                                    off_diag_weights=recon_cholesky_terms,
    #                                    local_connection_dist=supn_model.local_connection_dist)
    if batch:
        return LogProb.sum(), LogProb
    else:
        return LogProb.sum()


def log_likelihood_x_dat_given_x(A,x_dat,x,sigma_noise,forward_op=forward_op):
    n_dat = x_dat.shape[1]
    #n_dat = A.shape[0]
    return -0.5*((forward_op(A,x)-x_dat)**2/(sigma_noise**2) - torch.log(sigma_noise)).sum(axis=[1,2])


def sig_exp_transform(sig_eps):
    #return sig_eps**2 + 0.000001
    return torch.exp(sig_eps)
    #return softclip((sig_eps+2),0,1)

def plot_image(x):
    x = x.detach().cpu().clamp(-1,1).numpy()
    plt.imshow(x.reshape(128,128), cmap='gray')
    plt.axis('off')
    plt.colorbar()
    return plt.gcf()

def plot_image(x):
    #create subplot with 1 row and 2 columns plotting x[0] and x[1] as 128,128
    #grey images
    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].imshow(x[0].cpu().detach().numpy().reshape(128,128), cmap='gray')
    axs[0].axis('off')
    axs[1].imshow(x[1].cpu().detach().numpy().reshape(128,128), cmap='gray')
    axs[1].axis('off')
    return plt.gcf()


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


def sparse_precision_to_dense_covariance(x_logvar, x_cholesky_terms, local_connection_dist):
    batch_size, _, height, width = x_logvar.shape
    
    # Create identity matrix
    identity = torch.eye(height * width).to(x_logvar.device)
    
    L_prec_T = get_prec_chol_as_sparse_tensor(log_diag_weights=x_logvar,
                                            off_diag_weights=x_cholesky_terms,
                                            local_connection_dist=local_connection_dist)
    L_prec_T = L_prec_T[0].to_dense()

    prec = L_prec_T.T @ L_prec_T

    V, D, _ = torch.linalg.svd(prec)
    D_inv = torch.diag(1.0 / D)
    D_inv[3800:] = 0.0
    prec_inv = V @ D_inv @ V.T
    covariance = prec_inv

    #covariance = torch.linalg.solve(prec,identity.double())
    
    eigs_cov = torch.linalg.eigvals(covariance)
    eigs_cov = eigs_cov.real.sort(descending=True).values
    #plt.semilogy(eigs_cov.detach().cpu().numpy())
    plt.plot(eigs_cov.detach().cpu().numpy())
    plt.title(f'Eigenvalues of covariance matrix')

    # plt.imshow(L_prec_T.detach().cpu().numpy()[:200,:200])
    # plt.imshow(prec)
    # Apply sparse solve
    #supn_solver = SUPNSolver(x_logvar, x_cholesky_terms,local_connection_dist=2)

    #dense_covariance = supn_solver.solve_with_precision(-x_logvar, x_cholesky_terms, identity)
    
    # Reshape to standard covariance matrix form
    
    return covariance

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


#%% Library of settings and their corresponding experiment names
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
mask_prop = 0.025
im_ind = 5


if im_ind == 5:
    dataset = CustomDataset(train_data)
else:
    dataset = CustomDataset(test_data)

x_true = dataset[im_ind][0].to(device).unsqueeze(0)
settings_str =  f'{mask_type}_{mask_prop}_{im_ind}'
experiment_name = experiment_lib[settings_str]
filename = f'{settings_str}_map_run_{experiment_name}'
qz_pars = torch.load(f'post_estimation/checkpoints/separable_variational/middle_{mask_prop}_{im_ind}_map_run_{experiment_name}/60000/qz_pars.pth')
supn_model = torch.load(f'post_estimation/checkpoints/separable_variational/middle_{mask_prop}_{im_ind}_map_run_{experiment_name}/60000/supn_model.pth')

#%%
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

coord = [[80,40]]


p2 = torch.zeros(1,1,2,128,128).cuda()
clamp = [-0.015,0.015]
clamp2 = [-0.04,0.04]
for row_n, col_n in coord:
    rhs = torch.zeros(2*128**2).reshape(1,1,2,128,128)
    rhs[0,0,0,row_n,col_n] = 10
    rhs = rhs.to(device)

    p1 = prior_dist.chol_cov_mult(rhs, solve_with_upper=True)
    p2 = p2 + prior_dist.chol_cov_mult(p1, solve_with_upper=False)
    p2[0,0,0,row_n,col_n] = 0

pri_corr = p2

norm = mcolors.TwoSlopeNorm(vmin=clamp[0], vcenter=0, vmax=clamp[1])
plt.imshow(pri_mean[0,0].detach().cpu().numpy(),cmap='gray')
plt.imshow(p2[0,0,0].detach().cpu().clamp(clamp[0],clamp[1]).numpy(),cmap = 'bwr',norm=norm,alpha = 0.5)

norm2 = mcolors.TwoSlopeNorm(vmin=clamp2[0], vcenter=0, vmax=clamp2[1])
plt.imshow(pri_mean[0,0].detach().cpu().numpy(),cmap='gray')
plt.imshow(p2[0,0,1].detach().cpu().clamp(clamp2[0],clamp2[1]).numpy(),cmap = 'bwr',norm=norm2,alpha = 0.5)


fig, ax = plt.subplots(2, 4, figsize=(20, 10))

ax[0, 0].imshow(x_true[0, 0].detach().cpu().numpy(), cmap='gray')
ax[0, 0].axis('off')

ax[1, 0].imshow(x_true[0, 1].detach().cpu().numpy(), cmap='gray')
ax[1, 0].axis('off')

ax[0, 1].imshow(pri_mean[0, 0].detach().cpu().numpy(), cmap='gray')
ax[0, 1].axis('off')

ax[1, 1].imshow(pri_mean[0, 1].detach().cpu().numpy(), cmap='gray')
ax[1, 1].axis('off')

ax[0, 2].imshow(pri_resid[0, 0,0].detach().cpu().numpy(), cmap='gray')
ax[0, 2].axis('off')

ax[1, 2].imshow(pri_resid[0, 0,1].detach().cpu().numpy(), cmap='gray')
ax[1, 2].axis('off')

ax[0, 3].imshow(pri_mean[0, 0].detach().cpu().numpy(), cmap='gray')
ax[0, 3].imshow(p2[0, 0, 0, :, :].detach().cpu().clamp(clamp[0], clamp[1]).numpy(), cmap='bwr', norm=norm, alpha=0.5)
ax[0, 3].axis('off')

ax[1, 3].imshow(pri_mean[0, 1].detach().cpu().numpy(), cmap='gray')
ax[1, 3].imshow(p2[0, 0, 1, :, :].detach().cpu().clamp(clamp2[0], clamp2[1]).numpy(), cmap='bwr', norm=norm2, alpha=0.5)
ax[1, 3].axis('off')

#in ax[0,3] plot a white star at coord
ax[0, 3].plot(coord[0][1],coord[0][0],'w*',markersize=7)

#p3 = p2
#p3[0,0,:,-3:,:] = p3[0,0,:,-6:-3,:]
plt.tight_layout()

if not os.path.exists('figures'):
    os.makedirs('figures')
plt.savefig(f'figures/prior_correlations.pdf')
plt.show()




# %%
