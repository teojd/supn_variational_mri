import sys
sys.path.append('../')

import torch
from testing_utils import make_supn_weights, get_test_data
from supn_data import SUPNData
from supn_distribution import SUPN
from torch.distributions import MultivariateNormal
from sparse_precision_cholesky import apply_sparse_chol_rhs_matmul, get_prec_chol_as_sparse_tensor, \
    coo_to_supn_sparse


# testing parameters
local_connection_dist = 4 #note that local_connection_dist < min(im_size_w, im_size_h)/2 is required for filters to be valid
im_size_w = 10
im_size_h = im_size_w 
num_supn_models = 1
num_data_batches = 10
RANDOM_SEED = 42

RANDOM_DATA_SEED = 43
requires_grad = True

# Start by generating a SUPNData object containing a set of supn parameters
# [num_supn_dists, channels, w, h]
supn_prec_cholesky = make_supn_weights(local_connection_dist, im_size_w, im_size_h, RANDOM_SEED, num_supn_models)
mean = supn_prec_cholesky[0]
log_diag_weights = supn_prec_cholesky[1]
off_diag_weights = supn_prec_cholesky[2]


# Generate some test data [num_batches, num_supn_dists, channels, w, h] 
# #  (can be used as RHS, mean, x, etc.)
data = get_test_data(num_data_batches, num_supn_models, im_size_w, im_size_h, RANDOM_DATA_SEED, requires_grad)

print(f"mean shape: {mean.shape}")
print(f"data shape: {data.shape}")
print(f"log_diag_weights shape: {log_diag_weights.shape}")
print(f"off_diag_weights shape: {off_diag_weights.shape}")

# Create a SUPNData object from the supn parameters
supn_data = SUPNData(mean=mean, 
                     log_diag=log_diag_weights, 
                     off_diag=off_diag_weights, 
                     local_connection_dist=local_connection_dist)

# Create torch.distributions.SUPN object from the SUPNData object
supn_dist = torch.distributions.SUPN(supn_data)

# Add some testing functions here to check that the distribution object is working as
# expected. 
def test_mat_vec_mult(supn_dist, data):
    """
    Test that the matvec product of the SUPN distribution is the same as the
    dense matrix vector product

    Parameters
    ----------
    supn_dist : torch.distributions.SUPN
    SUPN distribution object
    data : torch.Tensor
    Data tensor to use for the test
    """
    supn_mat_vec_prod = apply_sparse_chol_rhs_matmul(dense_input = data,
                                log_diag_weights = supn_dist.supn_data.log_diag,
                                off_diag_weights = supn_dist.supn_data.off_diag,
                                local_connection_dist = supn_dist.supn_data.local_connection_dist,
                                use_transpose = True, 
                                use_3d = False)


    coo_prec_chol = get_prec_chol_as_sparse_tensor(log_diag_weights = supn_dist.supn_data.log_diag,
                                    off_diag_weights = supn_dist.supn_data.off_diag,
                                    local_connection_dist = supn_dist.supn_data.local_connection_dist,
                                    use_transpose = True,
                                    use_3d = False)

    dense_prec_chol = coo_prec_chol.to_dense()
    dense_mat_vec_prod_list = []
    for i in range(dense_prec_chol.shape[0]):
        dense_mat_vec_prod = torch.matmul(dense_prec_chol[i], data.reshape(data.shape[0],-1)[i])
        dense_mat_vec_prod_list.append(dense_mat_vec_prod)
    dense_mat_vec_prod = torch.stack(dense_mat_vec_prod_list).reshape(supn_mat_vec_prod.shape)
    assert torch.allclose(supn_mat_vec_prod, dense_mat_vec_prod)
    print("Mat-vec mult test passed")


def test_supn_sparse_torch_sparse_conversion(supn_dist):
    '''
    Test that the supn formatted precision cholesky is correctly converted to a
    sparse tensor, and that the sparse tensor is correctly converted back to a
    the supn format.

    Parameters
    ----------
    supn_dist : torch.distributions.SUPN
    SUPN distribution object with prespecified precision cholesky parameters
    '''
    # convert from supn format to sparse tensor
    coo_prec_chol_0 = get_prec_chol_as_sparse_tensor(log_diag_weights = supn_dist.supn_data.log_diag,
                                    off_diag_weights = supn_dist.supn_data.off_diag,
                                    local_connection_dist = supn_dist.supn_data.local_connection_dist,
                                    use_transpose = True,
                                    use_3d = False)

    # convert from sparse tensor to supn format
    supn_log_diag_weights, supn_off_diag_weights = coo_to_supn_sparse(coo_prec_chol_0)

    #note supn_dist.supn_data.off_diag will not be the same as supn_off_diag_weights
    #at the indices where supn_off_diag_weights is zero. These entries correspond to edges of the image 
    #where there is no connection and are ignored by the filters used in solve/matmul.

    #Test by first zeroing out these elements in supn_dist.supn_data.off_diag that are ignored
    supn_dist.supn_data.off_diag = supn_dist.supn_data.off_diag * (supn_off_diag_weights != 0)
    
    assert torch.allclose(supn_dist.supn_data.log_diag, supn_log_diag_weights)
    assert torch.allclose(supn_dist.supn_data.off_diag, supn_off_diag_weights)

    # could also do another conversion to coo and check that the coo is the same as the original coo
    coo_prec_chol_1 = get_prec_chol_as_sparse_tensor(log_diag_weights = supn_dist.supn_data.log_diag,
                                    off_diag_weights = supn_dist.supn_data.off_diag,
                                    local_connection_dist = supn_dist.supn_data.local_connection_dist,
                                    use_transpose = True,
                                    use_3d = False)

    assert torch.allclose(coo_prec_chol_0.to_dense(), coo_prec_chol_1.to_dense())

    print("Sparse conversion test passed")

def test_log_prob_correctness(supn_dist,data):
    '''
    Test that the log_prob function of the SUPN distribution is correctly
    implemented
    Parameters
    ----------
    supn_dist : torch.distributions.SUPN
    SUPN distribution object
    '''
    # get the log_prob from the supn sparse precision cholesky
    log_prob = supn_dist.log_prob(data)
 
    # get sparse coo precision cholesky
    sparse_prec_chol = get_prec_chol_as_sparse_tensor(log_diag_weights = supn_dist.supn_data.log_diag,
                                                    off_diag_weights = supn_dist.supn_data.off_diag,
                                                    local_connection_dist = supn_dist.supn_data.local_connection_dist,
                                                    use_transpose = False,
                                                    use_3d = False)[0]

    # compute the precision matrix from the coo precision cholesky
    sparse_prec = torch.sparse.mm(sparse_prec_chol, sparse_prec_chol.T)

    # convert to dense precision matrix
    dense_prec = sparse_prec.to_dense()

    # construct and evaluate log prob with dense precision matrix
    dense_normal = MultivariateNormal(loc=supn_dist.mean.squeeze(1).reshape(-1),
                                        precision_matrix=dense_prec)
    log_prob_from_dense = dense_normal.log_prob(data.reshape([data.shape[0],-1]))

    assert torch.allclose(log_prob, log_prob_from_dense)
    print("Log prob correctness test passed")
    
# Run tests

# mat-vec mult test
test_mat_vec_mult(supn_dist, data[0]) # data[0] since this isn't implemented for 5d batches of data yet

# sparse conversion test (both supn->coo and coo->supn)
test_supn_sparse_torch_sparse_conversion(supn_dist)

# log prob correctness test with random data
test_log_prob_correctness(supn_dist, data.squeeze(1)) # data.squeeze(1).shape = [num_data_batches, 1, im_size_w, im_size_h]

# log prob correctness test with model sample data

def test_log_prob_correctness_samples():
    samples = supn_dist.sample(5) # Ordering of the final reshape needs to be checked
    test_log_prob_correctness(supn_dist, samples.squeeze(1)) # data.squeeze(1).shape = [num_data_batches, 1, im_size_w, im_size_h]

test_log_prob_correctness_samples()
# Basic checks

# some basic checks that the outputs look correct in terms of shape and type
def test_samples():
    samples = supn_dist.sample(num_data_batches) # Ordering of the final reshape needs to be checked
    print(f'samples shape for a batch of {num_data_batches} and {num_supn_models} supn models: {samples.shape}')
    assert(samples.shape == (num_data_batches, num_supn_models, 1, im_size_w, im_size_h))

test_samples()

# check likelihoods compute outputs the correct shapes
likelihoods = supn_dist.log_prob(data[0]) # same reason for data[0] here
print(f'likelihoods shape for a batch of {num_data_batches} and {num_supn_models} supn models: {likelihoods.shape}')


# check that the precision cholesky is correct shape too
prec_chol = supn_dist.precision_cholesky
print(f'precision cholesky shape: {prec_chol.shape}')
