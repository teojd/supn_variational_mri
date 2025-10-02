import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import configargparse
import toml
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import sys
if os.getcwd().endswith('post_estimation'):
    os.chdir('..')

import matplotlib
matplotlib.use('Agg')

def calculate_psnr(image1, image2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
        image1 (numpy.ndarray): First image with shape (height, width, channels).
        image2 (numpy.ndarray): Second image with shape (height, width, channels).

    Returns:
        float: PSNR value in decibels (dB).
    """
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match

    max_pixel_value = np.max([image1.max(), image2.max()])
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)
    return psnr

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
im_ind = 5


if im_ind == 5:
    dataset = CustomDataset(train_data)
elif im_ind == 22:
    dataset = CustomDataset(test_data)
else:
    print('Invalid image index')

x_true = dataset[im_ind][0].to(device).unsqueeze(0)



measurement_prop = 0.025

supn_samples025 = np.load(f'post_estimation/samples/supn_samples_{measurement_prop}.npy')
dps_samples025 = np.load(f'post_estimation/samples/diff_samples_{measurement_prop}.npy')

measurement_prop = 0.4

supn_samples4 = np.load(f'post_estimation/samples/supn_samples_{measurement_prop}.npy')
dps_samples4 = np.load(f'post_estimation/samples/diff_samples_{measurement_prop}.npy')

images_top_row = [supn_samples025[6,0,0], supn_samples025[7,0,0], supn_samples025[2,0,0], dps_samples025[0,0,0], dps_samples025[1,0,0], dps_samples025[2,0,0]]
images_second_row = [supn_samples025[0,0,1], supn_samples025[1,0,1], supn_samples025[2,0,1], dps_samples025[0,0,1], dps_samples025[1,0,1], dps_samples025[2,0,1]]
images_third_row = [supn_samples4[0,0,0], supn_samples4[1,0,0], supn_samples4[2,0,0], dps_samples4[0,0,0], dps_samples4[1,0,0], dps_samples4[2,0,0]]
images_fourth_row = [supn_samples4[0,0,1], supn_samples4[1,0,1], supn_samples4[2,0,1], dps_samples4[0,0,1], dps_samples4[1,0,1], dps_samples4[2,0,1]]
images = [images_top_row, images_second_row, images_third_row, images_fourth_row]



#%%
# Grid size (rows x cols)
rows, cols = 2, 3  # Adjust grid size as needed

# Image size
img_size = 128

# Rectangle parameters (same for all images)
rect_x, rect_y, rect_w, rect_h = 25, 50, 20, 20  # Adjust as needed

rect_x2, rect_y2, rect_w2, rect_h2 = 63, 80, 30, 30  # Adjust as needed


# Create figure

cols = len(images[0])
rows = len(images)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))  # Adjust figure size

for i in range(rows):
    for j in range(cols):
        ax = axes[i, j]

        # Generate a random grayscale image
        image = images[i][j]
        # calculate PSNR to the true image
        if i%2 == 0:
            real_part = torch.tensor(image).unsqueeze(0)
            imag_part = torch.tensor(images[i+1][j]).unsqueeze(0)
            im4psnr = torch.concat((real_part, imag_part), dim=0)
            psnr_value = calculate_psnr(im4psnr.numpy(), x_true[0].cpu().numpy())
        vmin, vmax = image.min(), image.max()
        # Extract the enclosed pixels
        if i%2 == 0:
            zoomed_region = image[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]
        else:
            zoomed_region = image[rect_y2:rect_y2 + rect_h2, rect_x2:rect_x2 + rect_w2]
        # Show the original image
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if i%2 == 0: # display PSNR value on the image in the bottom left corner
            ax.text(0.07, 0.05, f'PSNR: {psnr_value:.2f}dB', color='white', fontsize=9,
                    transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

        # Add a red rectangle
        if i%2 == 0:
            if j<3:
                rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h, linewidth=2, edgecolor='blue', facecolor='none')
            else:
                rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h, linewidth=2, edgecolor='red', facecolor='none')
        else:
            if j<3:
                rect = patches.Rectangle((rect_x2, rect_y2), rect_w2, rect_h2, linewidth=2, edgecolor='blue', facecolor='none')
            else:
                rect = patches.Rectangle((rect_x2, rect_y2), rect_w2, rect_h2, linewidth=2, edgecolor='red', facecolor='none')

        ax.add_patch(rect)

        # Inset axes for zoomed region (adjust based on grid position)
        inset_size = 0.12  # Size of inset relative to figure
        inset_x = 0.08 + j * (1.0 / cols)  # Adjust horizontally
        inset_y = 0.9 - i * (1.0 / rows)  # Adjust vertically

        inset_ax = fig.add_axes([inset_x, inset_y, inset_size, inset_size])  
        inset_ax.imshow(zoomed_region, cmap='gray', vmin=vmin, vmax=vmax)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        # Set title for each subplot
plt.tight_layout()
# add headers to columns 2 and 5 saying 'supn' and 'dps'. Move the headers up so
# add these as xlabels on the and then position them so they are in the middle
# of the columns. Make them bigger
axes[0,1].set_xlabel('Ours')
axes[0,4].set_xlabel('DPS')
axes[0,1].xaxis.set_label_coords(0.5, 1.3)
axes[0,4].xaxis.set_label_coords(0.5, 1.3)
axes[0,1].xaxis.label.set_size(15)
axes[0,4].xaxis.label.set_size(15)


# add row labels on the left hand side between rows 1 and 2 saying '0.025' and between rows 3 and 4
# saying '0.4'. Make sure these labels are between rows and not on the rows
# themselves
axes[0,0].set_ylabel('2.5% Real')
axes[1,0].set_ylabel('2.5% Complex')
axes[2,0].set_ylabel('40% Real')
axes[3,0].set_ylabel('40% Complex')
axes[0,0].yaxis.label.set_size(15)
axes[1,0].yaxis.label.set_size(15)
axes[2,0].yaxis.label.set_size(15)
axes[3,0].yaxis.label.set_size(15)


if not os.path.exists('figures'):
    os.makedirs('figures')
plt.savefig(f'figures/dps_comparison_plot.pdf',bbox_inches='tight')
plt.show()


# %%
