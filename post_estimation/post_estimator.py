#%%
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

sys.path.append('../supn_train')
sys.path.append('..')


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
from supn_base.supn_data import SUPNData

import wandb
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
    return -torch.sum(0.5*sig_eps**2)

def log_prior_x_given_z(x,z,supn_model,batch = False):
    if x.shape[0] > 1:
        batch = True
    supn_dist = supn_model.decode(z.T)
    #recon_logvar = softclip(-recon_logvar)
    LogProb = supn_dist.log_prob(x)
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
#%%
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt = parse_config(default_config_path = 'recon_configs/config6')

    #log_wandb = opt['wandb']['log_wandb']
    dataset = opt['data']['dataset'] 
    image_size = opt['data']['image_size']
    num_channels = opt['data']['num_channels']
    device = opt['general']['device']
    latent_dimension_size = opt['model']['latent_dimension_size']
    local_connection_dist = opt['model']['local_connection_dist']
    use_attention = opt['model']['use_attention']
    use_group_norm = opt['model']['use_group_norm']
    init_decoder_var = torch.tensor(opt['model']['init_decoder_var'])
    measurement_prop = opt['recon_pars']['measurement_prop']
    num_var_samples = opt['recon_pars']['num_var_samples']
    test_set = opt['recon_pars']['test_set']
    im_ind = opt['recon_pars']['im_ind']
    noise_sd = opt['recon_pars']['noise_sd']
    mask_type = opt['recon_pars']['mask_type']

    supn_model = VAE(image_size = image_size,
                    latent_dim = latent_dimension_size,
                    local_connection_dist = local_connection_dist,
                    num_channels = num_channels,
                    use_attention = use_attention,
                    use_group_norm = use_group_norm,
                    init_decoder_var = init_decoder_var,
                    ).to(device)
    model_path = '../supn_train/checkpoints/pretrained.pth'
    
    supn_model.load_state_dict(torch.load(model_path))

    # Initialize wandb
    experiment_name = f"map_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if log_wandb:
        wandb.init(project="vae_supn_map", name=experiment_name)

    checkpoint_dir = f'checkpoints/separable_variational'

    # if there isn't a experiment_name folder in the checkpoint_dir folder, create it
    os.makedirs(f'{checkpoint_dir}/{mask_type}_{measurement_prop}_{im_ind}_{experiment_name}', exist_ok=True)

    # Training loops
    epoch = 0

    # Load the dataset
    train_data = np.load(f'../fastMRI/complex_train_images/knee_fastMRI_train_128_cleaned_complex.npy').transpose(0,2,3,1).astype(np.float32)
    test_data = np.load(f'../fastMRI/complex_test_images/knee_fastMRI_test_128_cleaned_complex.npy').transpose(0,2,3,1).astype(np.float32)

    if test_set:
        dataset = CustomDataset(test_data)
    else:
        dataset = CustomDataset(train_data)

    #set seed for train_loader
    x_true = dataset[im_ind][0].to(device).unsqueeze(0)


    if log_wandb:
        fig = plot_image(x_true[0])
        wandb.log({"x_true": wandb.Image(fig)})

    torch.manual_seed(0)
    
    # Create the mask A
    if mask_type == 'middle':
        A = middle_inds(128,measurement_prop*128**2).int()
    elif mask_type == 'horizontal':
        A = horiz_inds(128,measurement_prop*128**2).int()

    # visualise the mask
    M = torch.zeros(128**2)
    M[A] = 1
    plt.imshow(M.cpu().detach().numpy().reshape(128,128), cmap='gray')
    plt.title(f'Centre mask proportion: {measurement_prop}')

    # Simulate data by applying the forward operator with the mask
    x_dat = forward_op(A,x_true) 
    x_dat = x_dat + noise_sd*torch.randn_like(x_dat)

    if log_wandb:
        wandb.config.update({"measurement_prop": measurement_prop})
        wandb.config.update({"noise_sd": noise_sd})
        wandb.config.update({"local_connection_dist": local_connection_dist})
        wandb.config.update({"im_ind": im_ind})
        wandb.config.update({"model": model_path})
        wandb.config.update({"test set": test_set})
        wandb.config.update({"mask_type": mask_type})
        wandb.config.update({"num_var_samples": num_var_samples})


    # simulate 32 samples from the z prior to find the best starting point
    z0_sample = torch.randn(32,latent_dimension_size).to(device)
    _,z0_log_prob = log_prior_x_given_z(x_true,z0_sample.T,supn_model,batch=True)

    # find argmax of best_x0_log_prob
    best_ind = torch.argmax(z0_log_prob)
    z = torch.nn.Parameter(z0_sample[best_ind].to(device).clone().reshape([-1,1]))
    
    # Initialize the log variance of the noise
    log_sig_eps = torch.nn.Parameter(2*torch.ones_like(x_dat).to(device))

    # Initialize the parameters for the variational posterior q(z)
    qz_mu = torch.nn.Parameter(z0_sample[best_ind].to(device).clone().reshape([-1,1]).to(device))
    qz_cov_chol = torch.nn.Parameter(torch.eye(latent_dimension_size,latent_dimension_size).to(device))
    qz_pars = [qz_mu,qz_cov_chol]

    # evaluate the noise standard deviation
    sig_eps = sig_exp_transform(log_sig_eps)

    optimizer_z = torch.optim.Adam(qz_pars + [log_sig_eps], lr=1e-3,maximize=True)
    num_epochs = 10000
    for epoch in range(epoch+1, num_epochs + epoch + 1):
        optimizer_z.zero_grad()
        qz_cov = torch.matmul(qz_cov_chol,qz_cov_chol.T).cuda() + 0.0001*torch.eye(latent_dimension_size).cuda()
        qz_dist = torch.distributions.MultivariateNormal(qz_mu.squeeze(1).cuda(),qz_cov.cuda())
        qz_sample = qz_dist.rsample([num_var_samples])
        lpz = log_prior_z(qz_sample)
        x_z = supn_model.decode(qz_sample).supn_data.mean
        sig_eps = sig_exp_transform(log_sig_eps)
        ll = log_likelihood_x_dat_given_x(A,x_dat,x_z,sig_eps)
        lps = log_prior_sig_eps(sig_eps)
        joint_log_post = ll + lpz
        lqz = qz_dist.log_prob(qz_sample)
        log_qpost_q = lqz
        elbo = (joint_log_post - log_qpost_q).mean() + lps
        elbo.backward()
        optimizer_z.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {joint_log_post.mean().item()}")
            if log_wandb:
                wandb.log({"elbo": elbo.item()})
                wandb.log({"loss": joint_log_post.mean().item()})
                wandb.log({"sig_eps": sig_eps.mean().item()})
        if epoch % 1000 == 0:
            fig = plot_image(x_z[0])
            if log_wandb:
                wandb.log({"x_est": wandb.Image(fig)})
                plt.close()
                plt.plot(forward_op(A,x_true).detach().cpu()[0]-forward_op(A,x_z).detach().cpu()[0],alpha=0.4)
                wandb.log({"A*x - A*x_est": wandb.Image(plt.gcf())})
                plt.close()



    #log_sig_eps = torch.nn.Parameter(torch.tensor(log_sig_eps).to(device))


    q_x = supn_model.decode(z.T)
    q_x_mean = torch.nn.Parameter(q_x.supn_data.mean)
    q_x_log_diag = torch.nn.Parameter(q_x.supn_data.log_diag)
    q_x_off_diag = torch.nn.Parameter(q_x.supn_data.off_diag)
    q_x_cross_ch = torch.nn.Parameter(q_x.supn_data.cross_ch)
    post_supn_pars = [q_x_mean,q_x_log_diag,q_x_off_diag,q_x_cross_ch]

    q_supn_data = SUPNData(mean=q_x_mean,
                            log_diag=q_x_log_diag,
                            off_diag=q_x_off_diag,
                            cross_ch=q_x_cross_ch,
                            local_connection_dist=supn_model.local_connection_dist)
    q_supn_dist = SUPN(supn_data=q_supn_data)

    model_lr = 1e-3
    prior_lr = 1e-3
    sig_lr = 1e-3
    prior_pars = [{'params': qz_pars, 'lr': prior_lr}]
    std_pars = [{'params': log_sig_eps, 'lr': sig_lr}]
    supn_pars = [{'params': post_supn_pars, 'lr': model_lr}]


    optimizer = torch.optim.Adam(supn_pars + prior_pars + std_pars, maximize=True)
    num_epochs = 90000
    for epoch in range(epoch+1, num_epochs + epoch + 1):
        if epoch == 44000:
            model_lr = 1e-4
            prior_lr = 1e-4
            sig_lr = 1e-4
            prior_pars = [{'params': qz_pars, 'lr': prior_lr}]
            std_pars = [{'params': log_sig_eps, 'lr': sig_lr}]
            supn_pars = [{'params': post_supn_pars, 'lr': model_lr}]
            optimizer = torch.optim.Adam(supn_pars + prior_pars + std_pars, maximize=True)
        

        optimizer.zero_grad()
        qz_cov = torch.matmul(qz_cov_chol,qz_cov_chol.T).cuda() + 0.0001*torch.eye(latent_dimension_size).cuda()
        qz_dist = torch.distributions.MultivariateNormal(qz_mu.squeeze(1).cuda(),qz_cov.cuda())
        qz_sample = qz_dist.rsample([num_var_samples])
        lqz = qz_dist.log_prob(qz_sample)

        q_supn_data = SUPNData(mean=q_x_mean,
                               log_diag=q_x_log_diag,
                               off_diag=q_x_off_diag,
                               cross_ch=q_x_cross_ch,
                               local_connection_dist=supn_model.local_connection_dist)
        q_supn_dist.supn_data = q_supn_data


        q_supn_sample = q_supn_dist.sample(num_var_samples).squeeze(1)
        lqx = q_supn_dist.log_prob(q_supn_sample)

        sig_eps = sig_exp_transform(log_sig_eps)
        ll = log_likelihood_x_dat_given_x(A,x_dat,q_supn_sample,sig_eps)
        lpz = log_prior_z(qz_sample)
        lpx = log_prior_x_given_z(q_supn_sample,qz_sample.T,supn_model)[1]
        lps = log_prior_sig_eps(sig_eps)
        joint_log_post = ll + lpz + lpx
        log_qpost_q = lqx + lqz
        elbo = (joint_log_post - log_qpost_q).mean() + lps
        elbo.backward(retain_graph=True)
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {elbo.item()}")
            if log_wandb:
                wandb.log({"elbo": elbo.item()})
                wandb.log({"loss": joint_log_post.sum().item()})
                wandb.log({"sig_eps": sig_eps.mean().item()})
        if epoch % 100 == 0:
            fig = plot_image(q_supn_sample[0])
            if log_wandb:
                wandb.log({"x_est": wandb.Image(fig)})
                plt.close()
                plt.plot(forward_op(A,x_true).detach().cpu()[0]-forward_op(A,q_supn_sample[0].unsqueeze(0)).detach().cpu()[0],alpha=0.4)
                wandb.log({"A*x - A*x_est": wandb.Image(plt.gcf())})
                plt.close()
        if epoch % 1000 == 0:
            os.makedirs(f'{checkpoint_dir}/{mask_type}_{measurement_prop}_{im_ind}_{experiment_name}/{epoch}', exist_ok=True)
            torch.save(post_supn_pars, f'{checkpoint_dir}/{mask_type}_{measurement_prop}_{im_ind}_{experiment_name}/{epoch}/supn_model.pth')
            torch.save(qz_pars, f'{checkpoint_dir}/{mask_type}_{measurement_prop}_{im_ind}_{experiment_name}/{epoch}/qz_pars.pth')


    # Close wandb
    if log_wandb:
        wandb.finish()


    #plot_image(x[0])
    #plot_image(x_true)


if __name__ == "__main__":
    main()




