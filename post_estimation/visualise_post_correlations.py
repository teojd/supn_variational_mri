#%% Imports and set up functions

import sys
import os
import __main__ as main
# if the cwd is post_estimation, go up one directory
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
from matplotlib import pyplot as plt


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
experiment_lib = {'middle_0.025_22': '20250223_135259',
                  'middle_0.025_5': '20250223_205308',
                  'middle_0.05_22': '20250223_151551',
                  'middle_0.05_5': '20250223_221629',
                  'middle_0.1_22': '20250223_164020',
                  'middle_0.1_5': '20250223_233846',
                  'middle_0.2_22': '20250223_180319',
                  'middle_0.2_5': '20250224_010124',
                  'middle_0.4_22': '20250223_193058',
                  'middle_0.4_5': '20250224_022446'
                  }

mask_type = 'middle'
mask_prop = 0.025
im_ind = 22


if im_ind == 5:
    dataset = CustomDataset(train_data)
elif im_ind == 22:
    dataset = CustomDataset(test_data)
else:
    print('Invalid image index')

x_true = dataset[im_ind][0].to(device).unsqueeze(0)
settings_str =  f'{mask_type}_{mask_prop}_{im_ind}'
experiment_name = experiment_lib[settings_str]

qz_pars = torch.load(f'post_estimation/checkpoints/separable_variational/middle_{mask_prop}_{im_ind}_map_run_{experiment_name}/60000/qz_pars.pth')
supn_model = torch.load(f'post_estimation/checkpoints/separable_variational/middle_{mask_prop}_{im_ind}_map_run_{experiment_name}/60000/supn_model.pth')

#%% Load prior and sample from it
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

#%% plot the prior correlations by inverting the precision matrix

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

pri_corr = p2

norm = mcolors.TwoSlopeNorm(vmin=clamp[0], vcenter=0, vmax=clamp[1])
plt.imshow(pri_mean[0,0].detach().cpu().numpy(),cmap='gray')
plt.imshow(p2[0,0,0].detach().cpu().clamp(clamp[0],clamp[1]).numpy(),cmap = 'bwr',norm=norm,alpha = 0.5)



#%% Load the posterior models and sample from them
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
for prop in [0.025, 0.05, 0.1, 0.2, 0.4]:
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



#%% Compute the naive reconstructions using zero-filled IFFT
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


#%%
# make figure with 6x5 subplots showing:
# 1st row: prior residual, prior correlation, prior mean, prior sample, true image
# 2-6th row: posterior residual, posterior correlation, posterior mean, posterior sample, zero-filled reconstruction
clamp0 = [-0.06,0.06]
clamp = [-0.02,0.02]
clamp_1 = [-0.015,0.015]
norm0 = mcolors.TwoSlopeNorm(vmin=clamp0[0], vcenter=0, vmax=clamp0[1])
norm = mcolors.TwoSlopeNorm(vmin=clamp[0], vcenter=0, vmax=clamp[1])
norm_1 = mcolors.TwoSlopeNorm(vmin=clamp_1[0], vcenter=0, vmax=clamp_1[1])

fig, axs = plt.subplots(6, 5, figsize=(4.15*5, 25))
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

for i in range(5):
    #axs[i+1, 0+1].imshow(masks[i].cpu().detach().numpy().reshape(128,128), cmap='gray')
    #axs[i+1, 0+1].axis('off')
    axs[i+1, 1-1].imshow((post_samples[i][0,0,0] - post_means[i][0,0]).detach().cpu(), cmap='gray' )
    axs[i+1, 1-1].axis('off')
    axs[i+1, 3-1].imshow(post_means[i][0,0].detach().cpu().clamp(-1,1), cmap='gray')
    axs[i+1, 3-1].axis('off')
    axs[i+1, 4-1].imshow(post_samples[i][0,0,0].detach().cpu().clamp(-1,1), cmap='gray')
    axs[i+1, 4-1].axis('off')
    axs[i+1, 5-1].imshow(inversion_recons[i][:,:,0].real.detach().cpu().clamp(-1,1), cmap='gray')
    axs[i+1, 5-1].axis('off')
    axs[i+1, 2-1].imshow(post_corrs[i][0,0,0].real.detach().cpu().clamp(-1,1), cmap='gray',norm=norm)
    axs[i+1, 2-1].axis('off')

axs[i+1, 2-1].imshow(post_corrs[i][0,0,0].real.detach().cpu().clamp(-1,1), cmap='gray',norm=norm_1)

axs[0,0].set_title('Prior Residual', fontsize=20)
axs[0,1].set_title('Prior pixelwise correlation', fontsize=20)
axs[0,2].set_title('Prior mean', fontsize=20)
axs[0,3].set_title('Prior sample', fontsize=20)
axs[0,4].set_title('True image', fontsize=20)

axs[1,0].set_title('Posterior Residual', fontsize=20)
axs[1,1].set_title('Posterior pixelwise correlation', fontsize=20)
axs[1,2].set_title('Posterior mean', fontsize=20)
axs[1,3].set_title('Posterior sample', fontsize=20)
axs[1,4].set_title('Zero-filled reconstruction', fontsize=20)

# add row labels

axs[1,0].text(-0.1, 0.5, "2.5%", transform=axs[1,0].transAxes, 
             fontsize=20, verticalalignment='center', rotation=90)
axs[2,0].text(-0.1, 0.5, "5%", transform=axs[2,0].transAxes, 
             fontsize=20, verticalalignment='center', rotation=90)
axs[3,0].text(-0.1, 0.5, "10%", transform=axs[3,0].transAxes, 
             fontsize=20, verticalalignment='center', rotation=90)
axs[4,0].text(-0.1, 0.5, "20%", transform=axs[4,0].transAxes, 
             fontsize=20, verticalalignment='center', rotation=90)
axs[5,0].text(-0.1, 0.5, "40%", transform=axs[5,0].transAxes, 
             fontsize=20, verticalalignment='center', rotation=90)


axs[1, 0].set_ylabel(f'2.5%')
axs[2, 0].set_ylabel(f'5%')
axs[3, 0].set_ylabel(f'10%')
axs[4, 0].set_ylabel(f'20%')
axs[5, 0].set_ylabel(f'40%')

# add a red star to axs[0,1] at coords (80,40) and (60,30)
axs[0,1].scatter(39,58,marker='*',color='red',s=200)

fig.tight_layout()


#mak this directory if it does not exist
if not os.path.exists('figures'):
    os.makedirs('figures')
plt.savefig(f'figures/post_summary_{im_ind}_alt.pdf')


# %%
