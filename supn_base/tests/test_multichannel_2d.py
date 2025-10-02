import sys
sys.path.append('../')

import torch
from testing_utils import make_multi_channel_supn_weights, get_test_data
from supn_data import SUPNData, get_num_cross_channel_weights
from supn_distribution import SUPN
from sparse_precision_cholesky import apply_sparse_chol_rhs_matmul, get_prec_chol_as_sparse_tensor

# testing parameters
local_connection_dist = 1
im_size_w =7
im_size_h = 5
num_supn_models = 1
num_data_batches = 10
RANDOM_SEED = 42

RANDOM_DATA_SEED = 43
requires_grad = True

DTYPE = torch.float32

num_ch = 3

# Start by generating a SUPNData object containing a set of supn parameters
# [num_supn_dists, channels, w, h]
supn_prec_cholesky = make_multi_channel_supn_weights(local_connection_dist, im_size_w, im_size_h, num_ch, RANDOM_SEED, num_supn_models)
mean = supn_prec_cholesky[0]

log_diag_weights = supn_prec_cholesky[1]
off_diag_weights = supn_prec_cholesky[2]
cross_ch = supn_prec_cholesky[3]

# Generate some test data [num_batches, num_supn_dists, channels, w, h] 
# #  (can be used as RHS, mean, x, etc.)
data = get_test_data(num_data_batches, num_supn_models, im_size_w, im_size_h, random_seed=RANDOM_DATA_SEED, requires_grad=False, num_ch=num_ch)
print(f"mean shape: {mean.shape}")
print(f"data shape: {data.shape}")
print(f"log_diag_weights shape: {log_diag_weights.shape}")
print(f"off_diag_weights shape: {off_diag_weights.shape}")

# Create a SUPNData object from the supn parameters
supn_data = SUPNData(mean=mean, 
                     log_diag=log_diag_weights, 
                     off_diag=off_diag_weights, 
                     local_connection_dist=local_connection_dist,
                     cross_ch=cross_ch, use_3d=False)

# Create torch.distributions.SUPN object from the SUPNData object
supn_dist = torch.distributions.SUPN(supn_data)


# Add some tests here to check that the distribution object is working as
# expected. Currently just a consistency check using matmul and coo_prec_chol

def test_mat_vec_mult(supn_dist, data):
    supn_mat_vec_prod = apply_sparse_chol_rhs_matmul(dense_input = data,
                                log_diag_weights = supn_dist.supn_data.log_diag,
                                off_diag_weights = supn_dist.supn_data.off_diag,
                                local_connection_dist = supn_dist.supn_data.local_connection_dist,
                                use_transpose = True, 
                                use_3d = False,
                                cross_ch = supn_dist.supn_data.cross_ch)


    coo_prec_chol = get_prec_chol_as_sparse_tensor(log_diag_weights = supn_dist.supn_data.log_diag,
                                    off_diag_weights = supn_dist.supn_data.off_diag,
                                    local_connection_dist = supn_dist.supn_data.local_connection_dist,
                                    use_transpose = True,
                                    use_3d = False,
                                    cross_ch = supn_dist.supn_data.cross_ch)

    dense_prec_chol = coo_prec_chol.to_dense()
    dense_mat_vec_prod_list = [] 
    dense_mat_vec_prod = torch.matmul(dense_prec_chol[0], data.reshape(-1,1))

    for i in range(dense_prec_chol.shape[0]):
        dense_mat_vec_prod = torch.matmul(dense_prec_chol[i], data.reshape(data.shape[0],-1)[i])
        dense_mat_vec_prod_list.append(dense_mat_vec_prod)
    dense_mat_vec_prod = torch.stack(dense_mat_vec_prod_list).reshape(supn_mat_vec_prod.shape)
    assert torch.allclose(supn_mat_vec_prod, dense_mat_vec_prod)
    print("Mat-vec mult test passed")


def test_solver(supn_dist):
    import torch.autograd as autograd
    from torch.autograd import gradcheck
    from cholespy_solver import sample_zero_mean, supn_upper_cholesky_solve, supn_lower_cholesky_solve, supn_precision_solve

    def test_upper(log_diag, off_diag, cross_ch, noise_white):
        noise_supn_correlated = supn_upper_cholesky_solve(supn_dist.supn_solver, log_diag, off_diag, cross_ch, noise_white)
        return noise_supn_correlated
    
    n_samples = 10
    noise_white = torch.randn((supn_data.log_diag.shape[0], n_samples,)+supn_data.log_diag.shape[1:], device=supn_data.log_diag.device, dtype=supn_data.log_diag.dtype, requires_grad=False)
    
    gradcheck(test_upper, (supn_data.log_diag, supn_data.off_diag, supn_data.cross_ch, noise_white), eps=1e-4, atol=1e-5, check_undefined_grad=True, check_batched_grad=False)


    def test_lower(log_diag, off_diag, cross_ch, noise_white):
        noise_supn_correlated = supn_lower_cholesky_solve(supn_dist.supn_solver, log_diag, off_diag, cross_ch, noise_white)
        return noise_supn_correlated
    
    noise_white = torch.randn((supn_data.log_diag.shape[0], n_samples,)+supn_data.log_diag.shape[1:], device=supn_data.log_diag.device, dtype=supn_data.log_diag.dtype, requires_grad=False)
    
    gradcheck(test_lower, (supn_data.log_diag, supn_data.off_diag, supn_data.cross_ch, noise_white), eps=1e-4, atol=1e-5, check_undefined_grad=True, check_batched_grad=False)


    def test_precision(log_diag, off_diag, cross_ch, noise_white):
        noise_supn_correlated = supn_precision_solve(supn_dist.supn_solver, log_diag, off_diag, cross_ch, noise_white)
        return noise_supn_correlated
    
    noise_white = torch.randn((supn_data.log_diag.shape[0], n_samples,)+supn_data.log_diag.shape[1:], device=supn_data.log_diag.device, dtype=supn_data.log_diag.dtype, requires_grad=False)
    
    gradcheck(test_precision, (supn_data.log_diag, supn_data.off_diag, supn_data.cross_ch, noise_white), eps=1e-4, atol=1e-5, check_undefined_grad=True, check_batched_grad=False)

    print("Passed gradient checker for the multi-channel solver")


def test_samples():
    from cholespy_solver import sample_zero_mean
    samples = supn_dist.sample(num_data_batches) # Ordering of the final reshape needs to be checked
    #samples = sample_zero_mean(supn_data, num_data_batches, supn_dist.supn_solver)
    assert(samples.shape == (num_data_batches, num_supn_models, num_ch, im_size_w, im_size_h))

    print(f'samples shape for a batch of {num_data_batches} and {num_supn_models} supn models: {samples.shape}')

test_samples()

test_solver(supn_dist)

# data[0] since this isn't implemented for 5d batches of data yet
test_mat_vec_mult(supn_dist, data[0]) 

# check likelihoods compute outputs the correct shapes
# same reason for data[0] here
likelihoods = supn_dist.log_prob(data[0])
print(f'likelihoods shape for a batch of {num_data_batches} and {num_supn_models} supn models: {likelihoods.shape}')


# check that the precision cholesky is correct shape too
prec_chol = supn_dist.precision_cholesky
print(f'precision cholesky shape: {prec_chol.shape}')

exit()


