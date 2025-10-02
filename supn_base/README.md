# supn_base# supn_base

## What is this repo?

This repository is a minimalist, self-contained toolset for efficient solving of multivariate Gaussian distributions, including both covariance and mean information, based on sparse Cholesky decomposition. It is based on ideas described in the Structured Uncertainty Prediction Network (SUPN) paper ([Structured Uncertainty Prediction Networks](https://arxiv.org/abs/1802.07079)). It builds upon torch distributions and provides efficient methods for sparse handling of multivariate Gaussian distributions using Cholesky decomposition for precision matrices. It supports both CPU and GPU, making it easily integrable into your tensor pipeline.

## Features
- Efficient sparse Cholesky solver
- Integration with PyTorch pipelines
- Support for both CPU and GPU
- 
## Docker-Based Installation for Setting up Cholespy

See [SUPN Cholespy](https://github.com/ndfcampbell/supn_cholespy/tree/neill_dev_batched) for instructions on building the Docker image that contains the necessary classes and files. Replace `/its/home/ls749/` with your own home directory to mount it to `/home`:

```bash
# Clone the repository with the specified branch
git clone --recursive --branch neill_dev_batched git@github.com:ndfcampbell/supn_cholespy.git
cd supn_cholespy

# Build the Docker image (this step may take some time)
docker build -t supn_cholespy_image .

# Start the container in interactive mode with GPU support
# Bind your home directory to /home and set the shared memory size (default: 16GB). Adjust as needed.
docker run -it --shm-size=16gb --gpus all --rm -v rm -v /its/home/ls749/:/home -w /home supn_cholespy_image bash

export PYTHONPATH=~/supn_base:$PYTHONPATH

```
## Installation



TODO add later
## Main Classes and Functionalities

### SUPNData
Manages the data structure for SUPN, including methods for handling log-diagonal and off-diagonal weights.

### SUPN
Implements the SUPN distribution, providing methods for sampling, calculating log probabilities, and handling sparse precision Cholesky decomposition.

### Sparse Precision Cholesky
Contains functions to handle sparse precision matrices in Cholesky form, including:
- `convert_log_to_diag_weights`: Converts log weight values into positive diagonal values.
- `get_num_off_diag_weights`: Returns the number of off-diagonal entries required for a particular sparsity.
- `build_off_diag_filters`: Creates convolutional filters for off-diagonal components of the sparse Cholesky decomposition.

### SUPNSolver
Provides methods for solving linear systems using sparse Cholesky decomposition:
- `supn_upper_cholesky_solve`: Solves upper triangular systems.
- `supn_lower_cholesky_solve`: Solves lower triangular systems.
- `supn_precision_solve`: Solves systems using the precision matrix.



## To test it

You should be able to run the script `tests/tests.py`.


