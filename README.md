# Varitional Bayes MRI reconstruction with structured uncertainty distribution 

## Overview
This repo contains code and data associated with the paper 'Bayesian MRI Reconstruction with Structured Uncertainty Distributions'. 

A pretrained structured uncertainty prediction network (SUPN) prior is provided in `supn_train/checkpoints/pretrained.pth`. Alternatively this can be trained from scratch by running:
```bash
python supn_train/n_channel_train
``` 
This will train the model on 80% of the dataset for 600 epochs, saving a checkpoint every 100 epochs and also storing a copy of the models that attain the lowest test and train likelihood.


Fitted posterior approximations for various example images are provided in `post_estimation/checkpoints`. Various samples from these distributions, as well as the diffusion posterior sampler checkpoint and samples, are included to aid in visualising the results without retraining or resampling. 

All visualisations in the paper and supplement can be produced by running the `post_estimation/visualise_*.py` files from either the base directory of the `post_estimation` directory. These will be saved as pdf figures in the `figures` directory (which will be created if needed). 

Reconstruction posteriors can alternatively be fit from scratch by running, for example:
```bash
python post_estimator.py --config recon_configs/config1
```

The configs `recon_configs/configi` for $i = 1,2,3,...$ each reconstruct an example image and mask size shown in the paper, and can be edited to run alternative examples.

Experiment tracking is performed through Weights and Biases. This can be toggled off in the configuration (`supn_train_configs/fastMRI_complex_config`). This is useful for online visualisation of training and reconstructions, but can be toggled off in the relevant config files.


# Environment set up

The key dependency is the...

## SUPN_base class

The `supn_base` subdirectory contains the classes and methods for working with structured uncertainty distributions. This is packaged as a torch distribution with various properties and methods that can be used for VAE training or other settings. The version included here contains fast likelihood and sampling code on multichannel images, which is enough to use for VAE training and visualisation. If one wanted to use structured uncertainty distributions in their own project they would only need to modify their code to output the parameters of the distribution (the SUPN class defined in supn_base/supn_distribution.py). This class then gives functionality similar to that of a regular torch.distribution.

## Further requirements
Follow the installation procedure given in `supn_base/README.md` to install the SUPN base class. Once complete install the additional dependencies: 

```sh
pip install matplotlib wandb configargparse toml tqdm 
```
