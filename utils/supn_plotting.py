import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import torchvision
import matplotlib
import wandb

from torch.utils.data import DataLoader
from torchvision import transforms

matplotlib.use('Agg')

def imshow_transform(image):
    return image * 0.5 + 0.5

def visualise_encoded(model: nn.Module, 
                      device: torch.device,
                      data_loader: DataLoader,
                      train_set: bool =True,
                      epoch: int =0,
                      log_wandb: bool =False) -> None:
    model.eval()
    with torch.no_grad():

        # Load the test dataset
        uncropped_size = int(model.image_size/0.8)
        transform = transforms.Compose([
            transforms.Resize((uncropped_size, uncropped_size)),
            transforms.Grayscale(),
            transforms.CenterCrop(model.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        # Get a batch of training data
        data = next(iter(data_loader))
        data = data[0].to(device)

        # Get the reconstructed image
        supn_dist, _ = model.forward(data)
        batch_size, num_channels, height, width = supn_dist.mean.shape

        sample = supn_dist.sample(num_samples = 1)[0,0].permute(1,2,0)

        data0 = data[0].permute(1,2,0).cpu()
        sample0 = sample.detach().cpu()
        mean0 = supn_dist.mean[0].permute(1,2,0).detach().cpu()

        # visualise in grid
        if num_channels == 1 or num_channels == 3:
            fig1 = plt.figure(figsize=(10, 10))
            vmin, vmax = 0, 1 # Set consistent color scaling limits
            plt.subplot(2, 2, 1)
            plt.imshow(imshow_transform(data0.numpy()), vmin=vmin, vmax=vmax)
            plt.title(f'Original')
            plt.subplot(2, 2, 2)
            plt.imshow(imshow_transform(mean0), vmin=vmin, vmax=vmax)
            plt.title(f'Mean reconstruction')
            plt.subplot(2, 2, 3)
            plt.imshow(imshow_transform(-mean0 + sample0), vmin=vmin, vmax=vmax)
            plt.title(f'Noise sample')
            plt.subplot(2, 2, 4)
            plt.imshow(imshow_transform(sample0), vmin=vmin, vmax=vmax)
            plt.title(f'Mean + noise sample')
            plt.show()
            if log_wandb:
                if train_set:
                    wandb.log({'Train_Recon': wandb.Image(fig1)})
                else:
                    wandb.log({'Test_Recon': wandb.Image(fig1)})
        elif num_channels == 2:
            fig1 = plt.figure(figsize=(10, 10))
            vmin, vmax = 0, 1
            plt.subplot(2, 2, 1)
            plt.imshow(imshow_transform(data0[:,:,0].numpy()), vmin=vmin, vmax=vmax)
            plt.title(f'Original Channel 1')
            plt.subplot(2, 2, 2)
            plt.imshow(imshow_transform(mean0[:,:,0]), vmin=vmin, vmax=vmax)
            plt.title(f'Mean reconstruction Channel 1')
            plt.subplot(2, 2, 3)
            plt.imshow(imshow_transform(-mean0[:,:,0] + sample0[:,:,0]), vmin=vmin, vmax=vmax)
            plt.title(f'Noise sample Channel 1')
            plt.subplot(2, 2, 4)
            plt.imshow(imshow_transform(sample0[:,:,0]), vmin=vmin, vmax=vmax)
            plt.title(f'Mean + noise sample Channel 1')
            plt.show()
            fig2 = plt.figure(figsize=(10, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(imshow_transform(data0[:,:,1].numpy()), vmin=vmin, vmax=vmax)
            plt.title(f'Original Channel 2')
            plt.subplot(2, 2, 2)
            plt.imshow(imshow_transform(mean0[:,:,1]), vmin=vmin, vmax=vmax)
            plt.title(f'Mean reconstruction Channel 2')
            plt.subplot(2, 2, 3)
            plt.imshow(imshow_transform(-mean0[:,:,1] + sample0[:,:,1]), vmin=vmin, vmax=vmax)
            plt.title(f'Noise sample Channel 2')
            plt.subplot(2, 2, 4)
            plt.imshow(imshow_transform(sample0[:,:,1]), vmin=vmin, vmax=vmax)
            plt.title(f'Mean + noise sample Channel 2')
            plt.show()
            if log_wandb:
                if train_set:
                    wandb.log({'Train_Recon1': wandb.Image(fig1)})
                    wandb.log({'Train_Recon2': wandb.Image(fig2)})
                else:
                    wandb.log({'Test_Recon1': wandb.Image(fig1)})
                    wandb.log({'Test_Recon2': wandb.Image(fig2)})

    model.train()

def visualise_samples(model: nn.Module,
              device: torch.device,
              n_samples: int,
              epoch: int = None,
              log_wandb: bool = False,
              cov_pix: tuple = (15,15)) -> None:
    vmin, vmax = 0, 1
    model.eval()
    with torch.no_grad():
        z_sample = torch.randn(n_samples, model.latent_dim).to(device)
        supn_dist = model.decode(z_sample)
        num_channels, height, width = supn_dist.mean.shape[1:]
        mean_sample = supn_dist.mean#.view(n_samples, num_channels, model.image_size, model.image_size)
        diag_samples = supn_dist.supn_data.log_diag.exp()#.view(n_samples, num_channels, model.image_size, model.image_size)

        # Create grid of samples
        mean_grid = torchvision.utils.make_grid(mean_sample/2 +0.5, nrow=4)
        diag_grid = torchvision.utils.make_grid(diag_samples.sqrt(), nrow=4)

        if log_wandb:
            if num_channels ==2:
                wandb.log({
                    "Mean samples channel 1": wandb.Image(mean_grid[0]),
                    "Mean samples channel 2": wandb.Image(mean_grid[1]),
                    "Log diag choleskey samples channel 1": wandb.Image(diag_grid[0]),
                    "Log diag choleskey samples channel 2": wandb.Image(diag_grid[1])
                    })
            elif num_channels == 1 or num_channels == 3:
                wandb.log({
                    "Mean samples": wandb.Image(mean_grid),
                    "Log diag choleskey samples": wandb.Image(diag_grid)
                    })
        #plt.figure(figsize=(10, 5))
        #plt.subplot(1, 2, 1)
        #if num_channels == 1 or num_channels == 3:
        #    plt.imshow(imshow_transform(mean_grid.permute(1, 2, 0).cpu().numpy()), vmin=vmin, vmax=vmax)
        #else: #plot the first two channels in different supblots
        #    plt.subplot(1, 2, 1)
        #    plt.imshow(imshow_transform(mean_grid[0].permute(1, 2, 0).cpu().numpy()), vmin=vmin, vmax=vmax)
        #    plt.subplot(1, 2, 2)
        #    plt.imshow(imshow_transform(mean_grid[1].permute(1, 2, 0).cpu().numpy()), vmin=vmin, vmax=vmax)
        #plt.title('Samples')
        #plt.subplot(1, 2, 2)
        #plt.imshow(imshow_transform(diag_grid.permute(1, 2, 0).cpu().numpy()), vmin=vmin, vmax=vmax)
        #plt.title('Log diag cholesky')
        #plt.show()

        if model.local_connection_dist>0:
            supn_dist = model.decode(z_sample[0].reshape(1, -1))

            batch_size, _, height, width = supn_dist.supn_data.log_diag.shape

            # Create identity matrix
            # identity = torch.eye(height * width).to(supn_dist.log_diag.device)
            # ind = cov_pix[0]*model.image_size + cov_pix[1]
            # rhs = identity[ind].reshape(supn_dist.log_diag.shape)

            #dense_covariance_row = supn_linsolve_gpu(supn_dist.log_diag, supn_dist.off_diag, model.local_connection_dist, rhs)
            #dense_covariance_row_np = dense_covariance_row.detach().cpu().numpy().reshape([height,width])

            # plot x_recon, variance, and dense_covariance in one figure
            if num_channels == 1 or num_channels == 3:
                vmin, vmax = 0, 1 # Set consistent color scaling limits
                fig1 = plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(imshow_transform(supn_dist.mean[0, :, :, :].permute(1,2,0).detach().cpu().numpy()), vmin=vmin, vmax=vmax)
                plt.title('Reconstructed Image')
                plt.subplot(1, 3, 2)
                plt.imshow(supn_dist.supn_data.log_diag[0, 0, :, :].exp().sqrt().detach().cpu().numpy(), vmin=vmin, vmax=vmax)
                plt.title('Standard Deviation')
                plt.subplot(1, 3, 3)
                #plt.imshow(imshow_transform(dense_covariance_row_np)
                plt.title('Covariance row')
            elif num_channels == 2:
                vmin, vmax = 0, 1 # Set consistent color scaling limits
                fig1 = plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(imshow_transform(supn_dist.mean[0, 0, :, :].detach().cpu().numpy()), vmin=vmin, vmax=vmax)
                plt.title('Reconstructed Image Channel 1')
                plt.subplot(1, 3, 2)
                plt.imshow(supn_dist.mean[0, 1, :, :].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
                plt.title('Reconstructed Image Channel 2')
                plt.subplot(1, 3, 3)
                plt.imshow(supn_dist.supn_data.log_diag[0, 0, :, :].exp().sqrt().detach().cpu().numpy(), vmin=vmin, vmax=vmax)
            
            if log_wandb:
                wandb.log({"Reconstruction_Variance_Covariance": wandb.Image(fig1)})

            #plot samples 3 from the covariance matrix
            if num_channels == 1 or num_channels == 3:
                fig2 = plt.figure(figsize=(15, 5))
                sample = supn_dist.sample(num_samples = 3).squeeze(1)
                for i in range(3):
                    plt.subplot(1, 3, i + 1)
                    sample_np = sample[i].permute(1,2,0).detach().cpu().numpy()
                    plt.imshow(imshow_transform(sample_np))
                    plt.title(f'Sample {i + 1}')
            elif num_channels == 2:
                fig2 = plt.figure(figsize=(15, 10))
                sample = supn_dist.sample(num_samples = 3).squeeze(1)
                for i in range(3):
                    plt.subplot(2, 3, i + 1)
                    plt.imshow(imshow_transform(sample[i, 0, :, :].detach().cpu().numpy()), vmin=vmin, vmax=vmax)
                    plt.title(f'Sample {i + 1} Channel 1')
                    plt.subplot(2, 3, i + 4)
                    plt.imshow(imshow_transform(sample[i, 1, :, :].detach().cpu().numpy()), vmin=vmin, vmax=vmax)
                    plt.title(f'Sample {i + 1} Channel 2')

            if log_wandb:
                wandb.log({'Covariance_Samples': wandb.Image(fig2)})

            #plot samples 3 from the supn distribution
            if num_channels == 1 or num_channels == 3:
                fig3 = plt.figure(figsize=(15, 5))
                for i in range(3):
                    plt.subplot(1, 3, i + 1)
                    sample_np = sample[i].permute(1,2,0).detach().cpu().numpy()
                    plt.imshow(imshow_transform(supn_dist.mean[0].permute(1,2,0).detach().cpu() + sample_np), vmin=vmin, vmax=vmax)
                    plt.title(f'Sample {i + 1}')
            elif num_channels == 2:
                fig3 = plt.figure(figsize=(15, 10))
                for i in range(3):
                    plt.subplot(2, 3, i + 1)
                    plt.imshow(imshow_transform(supn_dist.mean[0, 0, :, :].detach().cpu().numpy() + sample[i, 0, :, :].detach().cpu().numpy()), vmin=vmin, vmax=vmax)
                    plt.title(f'Sample {i + 1} Channel 1')
                    plt.subplot(2, 3, i + 4)
                    plt.imshow(imshow_transform(supn_dist.mean[0, 1, :, :].detach().cpu().numpy() + sample[i, 1, :, :].detach().cpu().numpy()), vmin=vmin, vmax=vmax)
                    plt.title(f'Sample {i + 1} Channel 2')

            if log_wandb:
                wandb.log({"Reconstruction_Variance_Covariance": wandb.Image(fig1),
                        "Covariance_Samples": wandb.Image(fig2),
                        "Reconstructed_Samples": wandb.Image(fig3)
                        })



    model.train()