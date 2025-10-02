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


import toml
import configargparse
import numpy as np
import wandb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from datetime import datetime

from utils.losses import log_likelihood_loss, VAE_loss
from models.supn_models.colour import VAE
from utils.supn_plotting import visualise_encoded, visualise_samples

#%%

# Used when a new phase of training begins
def update_optimiser(optimiser: torch.optim.Optimizer,
                     lr: float,
                     trainable_params: list) -> torch.optim.Optimizer:
    """
    Update the optimizer's learning rate and trainable parameters.
    
    Args:
        optimiser (torch.optim.Optimizer): The optimizer to update.
        lr (float): The new learning rate to set.
        trainable_params (list): A list of the trainable parameters to update.
    
    Returns:
        torch.optim.Optimizer: The updated optimizer.
    """
    optimiser = torch.optim.Adam(trainable_params, lr) 
    return optimiser

'''
would like to be able to update the params and lr of the optimiser
without having to reinitialise the optimiser...
def update_optimiser(optimiser,lr,trainable_params):
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr
        for param in param_group:
            param.requires_grad = False
    for param_group in trainable_params:
        param_group.requires_grad = True
    return optimiser
'''




def train(model: nn.Module,
          device: torch.device,
          train_loader: DataLoader,
          loss_fn: callable,
          optimizer: torch.optim.Optimizer,
          epoch: int,
          experiment_name: str,
          log_wandb: bool = False,
          best_train_loss = float('inf')) -> None:
    """
    Function to train the SUPN model on the train dataset for one epoch.
    
    Args:
        model (nn.Module): The SUPN model to train.
        device (torch.device): The device to use for training.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        loss_fn (callable): The loss function to use for training (log likelihood or VAE loss).
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        epoch (int): The current training epoch.
        experiment_name (str): The name of the current experiment.
        log_wandb (bool, optional): Whether to log the training metrics to Weights & Biases. Defaults to False.
        cov_pix (tuple, optional): The indices of the pixel to visualize the covariance of. Defaults to (15, 15).
    
    Returns:
        None
    """
    model.train()
    train_loss = 0
    NLL = 0
    progress_bar = tqdm(train_loader, desc=f'Train Epoch: {epoch}')
    for batch_idx, (data, _) in enumerate(progress_bar):
        data = data.to(device)
        optimizer.zero_grad()
        batch_loss, batch_NLL = loss_fn(data, model)
        batch_loss.backward()
        train_loss += batch_loss.item()
        NLL += batch_NLL.item()
        optimizer.step()
        progress_bar.set_postfix({'Loss': f'{batch_loss.item() / len(data):.6f}'})
        
        # Log loss to wandb
        if log_wandb:
            wandb.log({'Loss/train': batch_loss.item() / len(data)})
    
    avg_loss = train_loss / len(train_loader.dataset)
    avg_NLL = NLL / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} Average logdiag: {model.forward(data)[0].supn_data.log_diag.mean():.4f}')
    
    # Log average loss for the epoch
    if log_wandb:
        wandb.log({'Loss/train_epoch': avg_loss, 'epoch_step': epoch})
        wandb.log({'Likelihood/train_epoch': avg_NLL, 'epoch_step': epoch})
    
    # Save model checkpoint if it's the best loss so far
    if epoch == 1 or avg_loss < best_train_loss:
        best_train_loss = avg_loss
        torch.save(model.state_dict(), f'checkpoints/supn_train_checkpoints/best_train_{experiment_name}.pth')

    del NLL, train_loss, batch_loss, batch_NLL, avg_loss, avg_NLL
    torch.cuda.empty_cache()

    return best_train_loss 


def validate(model: nn.Module,
          device: torch.device,
          test_loader: DataLoader,
          loss_fn: callable,
          epoch: int,
          experiment_name: str,
          mean: bool = False,
          log_wandb: bool = False,
          best_test_loss = float('inf')) -> None:

    """
    Function to validate the SUPN model on the test dataset at the end of an epoch.
    
    Args:
        model (nn.Module): The SUPN model to test.
        device (torch.device): The device to use for testing.
        test_loader (torch.utils.data.DataLoader): The testing data loader.
        epoch (int): The current testing epoch.
        experiment_name (str): The name of the current experiment.
        mean (bool, optional): (Currently unused probably broken) Supposedly allows a custom mean to be set if desired so covariance only is learned. Defaults to False.
        loss_fn (callable): The loss function to use for validation (log likelihood or VAE loss).
        log_wandb (bool, optional): Whether to log the testing metrics to Weights & Biases. Defaults to False.
        cov_pix (tuple, optional): The indices of the pixel to visualize the covariance of. Defaults to (15, 15).
    
    Returns:
        None
    """
    model.eval()
    with torch.no_grad():
        test_loss = 0
        NLL = 0
        progress_bar = tqdm(test_loader, desc=f'Train Epoch: {epoch}')
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            batch_loss, batch_NLL = loss_fn(data, model)
            test_loss += batch_loss.item()
            NLL += batch_NLL.item()
            progress_bar.set_postfix({'Loss': f'{batch_loss.item() / len(data):.6f}'})
            
            # Log loss to wandb
            if log_wandb:
                wandb.log({'Loss/test': batch_loss.item() / len(data)})

        avg_loss = test_loss / len(test_loader.dataset)
        avg_NLL = NLL / len(test_loader.dataset)
        print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} Average logdiag: {model.forward(data)[0].supn_data.log_diag.mean():.4f}')
        
        # Log average loss for the epoch
        if log_wandb:
            wandb.log({'Loss/test_epoch': avg_loss, 'epoch_step': epoch})
            wandb.log({'Likelihood/test_epoch': avg_NLL, 'epoch_step': epoch})


    if epoch == 1 or avg_loss < best_test_loss:
        best_test_loss = avg_loss
        torch.save(model.state_dict(), f'checkpoints/supn_train_checkpoints/best_test_{experiment_name}.pth')

    if epoch%50 == 0:
        test_supn = model(data)[0]
        samples = test_supn.sample(3)
        fig, ax = plt.subplots( nrows=3, ncols=4 )
        ax[0,0].imshow(torch.clamp(samples[0,0,0,:,:].detach().cpu(),-1,1))
        ax[0,1].imshow(torch.clamp(samples[0,0,1,:,:].detach().cpu(),-1,1))
        ax[0,2].imshow(torch.clamp(samples[0,1,0,:,:].detach().cpu(),-1,1))
        ax[0,3].imshow(torch.clamp(samples[0,1,1,:,:].detach().cpu(),-1,1))
        ax[1,0].imshow(torch.clamp(samples[1,0,0,:,:].detach().cpu(),-1,1))
        ax[1,1].imshow(torch.clamp(samples[1,0,1,:,:].detach().cpu(),-1,1))
        ax[1,2].imshow(torch.clamp(samples[1,1,0,:,:].detach().cpu(),-1,1))
        ax[1,3].imshow(torch.clamp(samples[1,1,1,:,:].detach().cpu(),-1,1))
        ax[2,0].imshow(torch.clamp(samples[2,0,0,:,:].detach().cpu(),-1,1))
        ax[2,1].imshow(torch.clamp(samples[2,0,1,:,:].detach().cpu(),-1,1))
        ax[2,2].imshow(torch.clamp(samples[2,1,0,:,:].detach().cpu(),-1,1))
        ax[2,3].imshow(torch.clamp(samples[2,1,1,:,:].detach().cpu(),-1,1))
        fig.savefig('test2.png')
        if log_wandb:
            wandb.log({'Batch_sample_test': wandb.Image(fig)})

    del NLL, test_loss, batch_loss, batch_NLL, avg_loss, avg_NLL
    torch.cuda.empty_cache()

    return best_test_loss


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

# Custom dataset definition
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
    # Parse command line arguments
    opt = parse_config(default_config_path = 'supn_train_configs/fastMRI_complex_config')

    log_wandb = opt['wandb']['log_wandb']
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
    
    #log_wandb = False

    experiment_name = f"supn_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # make folder in checkpoints/experiment_name
    if not os.path.exists(f'checkpoints/{experiment_name}'):
        os.makedirs(f'checkpoints/supn_train_checkpoints/{experiment_name}')

    # Initialize wandb
    if log_wandb:
        wandb.init(project=project_name, name=experiment_name, config=opt)
        # log all hyperparameters and save code
        wandb.run.log_code(".")

    # Training loops
    epoch = 0

    # Load dataset...
    uncropped_resolution = int(image_size/0.8)
    

    if num_channels == 1:
        transform = transforms.Compose([
            transforms.Resize((uncropped_resolution, uncropped_resolution)),
            transforms.Grayscale(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
    elif num_channels == 2:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5), (0.5, 0.5))])
    elif num_channels == 3:
        transform = transforms.Compose([
            transforms.Resize((uncropped_resolution, uncropped_resolution)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #dataset = datasets.ImageFolder(root=f'datasets/{dataset}',
    #                                     transform=transform)
    
    #if n_data_points != 'all':
    #    dataset = torch.utils.data.Subset(dataset, range(n_data_points))
    

    # split dataset into train and test
    #torch.manual_seed(42)
    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])


    # set up dataloaders
    train_data = np.load(f'fastMRI/complex_train_images/knee_fastMRI_train_128_cleaned_complex.npy').transpose(0,2,3,1).astype(np.float32)
    test_data = np.load(f'fastMRI/complex_test_images/knee_fastMRI_test_128_cleaned_complex.npy').transpose(0,2,3,1).astype(np.float32)

    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    #define_model
    model = VAE(image_size = image_size,
                latent_dim = latent_dimension_size,
                local_connection_dist = local_connection_dist,
                num_channels = num_channels,
                use_attention = use_attention,
                use_group_norm = use_group_norm,
                init_decoder_var = init_decoder_var,
                ).to(device)

    #x = next(iter(train_loader))[0].cuda()
    #sup = model(x)[0]
    #ims = sup.sample(1)
    #plt.imshow(ims[0,0,0].reshape([128,128,1]).detach().cpu())
    #plt.savefig('test.png')

    best_train_loss = float('inf')
    best_test_loss = float('inf')


    # if a trained model is defined in the config, load it
    if load_trained_supn_model:
        model.load_state_dict(torch.load(supn_model_load_path, map_location=device))

    if log_wandb:
        wandb.watch(model)

    # separate the parameters into groups for separate optimisation
    mean_params = [{'params': model.params.encoder.parameters()},
        {'params': model.params.decoder_mean.parameters()}]

    dec_params = [{'params': model.params.decoder_mean.parameters()},
                  {'params': model.params.decoder_chol.parameters()},
                  {'params': model.scaling}]
    
    chol_params = [{'params': model.params.decoder_chol.parameters()},
                  {'params': model.scaling}]
    
    all_params = [{'params': model.params.encoder.parameters()},
                  {'params': model.params.decoder_mean.parameters()},
                  {'params': model.params.decoder_chol.parameters()},
                  {'params': model.scaling}]
    

    # define a dict to hold the parameters for each stage so they can be referenced in the config
    par_dict = {'mean': mean_params, 
                'dec': dec_params, 
                'chol': chol_params, 
                'all': all_params}
    
    # define a dict the loss functions for each stage so they can be referenced in the config
    loss_dict = {'VAE_loss': VAE_loss,
                 'log_likelihood_loss': log_likelihood_loss}

    # for each stage of train_schedule, update the optimiser as appropriate for the current schedule 
    # and train the model
    optimiser = torch.optim.Adam(mean_params, lr=1e-4)   
    stage_idx = 0 
    for stage in train_schedule:
        stage_idx += 1
        lr = stage['learning_rate']
        params = par_dict[stage['parameters']]
        num_epochs = stage['num_epochs']
        loss_type = loss_dict[stage['loss_type']]

        optimiser = update_optimiser(optimiser, lr, params)

        for epoch in range(epoch+1, epoch + num_epochs + 1):
            best_train_loss = train(model=model, 
                                    device = device,
                                    train_loader = train_loader,
                                    loss_fn = loss_type,
                                    optimizer = optimiser,
                                    epoch = epoch,
                                    experiment_name = experiment_name,
                                    log_wandb = log_wandb,
                                    best_train_loss = best_train_loss)

            best_test_loss = validate(model = model,
                                       device = device,
                                       test_loader = test_loader,
                                       loss_fn = loss_type,
                                       epoch = epoch,
                                       experiment_name = experiment_name,
                                       log_wandb = log_wandb,
                                       best_test_loss = best_test_loss)

            if epoch == 1 or epoch % 1 == 0:
            # Visualize and log images
                visualise_samples(model = model, 
                        device = device, 
                        n_samples = 16, 
                        epoch = epoch,
                        log_wandb=log_wandb,
                        cov_pix = cov_pix)
                        
                visualise_encoded(model = model, 
                                device = device,
                                data_loader = train_loader, 
                                train_set = True,
                                epoch = epoch, 
                                log_wandb = log_wandb)

                visualise_encoded(model = model, 
                                device = device,
                                data_loader = test_loader, 
                                train_set = False,
                                epoch = epoch, 
                                log_wandb = log_wandb)

            torch.cuda.empty_cache()

            if epoch % 100 == 0:
                torch.save(model.state_dict(), f'supn_train/checkpoints/supn_train_checkpoints/{experiment_name}/epoch_{epoch}.pth')

        # Save the trained model
        torch.save(model.state_dict(), f'supn_train/checkpoints/supn_train_checkpoints/{experiment_name}/model_stage_{stage_idx}.pth')
        print(f'Stage {stage_idx} complete. Model saved.')
    # Close wandb
    if log_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()

