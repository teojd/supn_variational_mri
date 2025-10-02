import torch
from torch.autograd import Variable

from supn_data import get_num_off_diag_weights, get_num_cross_channel_weights


# device = 'cpu'
device = 'cuda'
dtype = torch.float64

RANDOM_SEED = 42
RANDOM_RHS_SEED = 43

if dtype == torch.float64:
    ASSERT_THRESHOLD = 1.0e-12
else:
    ASSERT_THRESHOLD = 1.0e-6

use_multiple_rhs = True


def make_supn_weights(local_connection_dist, im_size_w, im_size_h, random_seed, batch_size=1):
    num_off_diag_weights = get_num_off_diag_weights(local_connection_dist)

    torch.manual_seed(random_seed)
    log_diag_weights = Variable(
        0.5 * torch.randn((batch_size, 1, im_size_w, im_size_h), dtype=dtype, device=device) - 0.0,
        requires_grad=True)
    mean = Variable(
        0.5 * torch.randn((batch_size, 1, im_size_w, im_size_h), dtype=dtype, device=device) - 0.0,
        requires_grad=True)
    off_diag_weights = Variable(
        0.1 * torch.randn((batch_size, num_off_diag_weights, im_size_w, im_size_h), dtype=dtype, device=device),
        requires_grad=True)

    return mean, log_diag_weights, off_diag_weights

def make_multi_channel_supn_weights(local_connection_dist, im_size_w, im_size_h, num_ch, random_seed, batch_size):
    """
    Create a set of SUPN weights for a multi-channel model
    """
    num_off_diag_weights = get_num_off_diag_weights(local_connection_dist)
    num_cross_channel_weights = get_num_cross_channel_weights(num_ch)

    torch.manual_seed(random_seed)
    log_diag_weights = Variable(
        0.5 * torch.randn((batch_size, num_ch, im_size_w, im_size_h), dtype=dtype, device=device) - 0.0,
        requires_grad=True)
    mean = Variable(
        0.5 * torch.randn((batch_size, num_ch, im_size_w, im_size_h), dtype=dtype, device=device) - 0.0,
        requires_grad=True)
    off_diag_weights = Variable(
        0.1 * torch.randn((batch_size, num_ch * num_off_diag_weights, im_size_w, im_size_h), dtype=dtype, device=device),
        requires_grad=True)
    
    cross_channel_weights = Variable(
        0.1 * torch.randn((batch_size, num_cross_channel_weights, im_size_w, im_size_h), dtype=dtype, device=device),
        requires_grad=True)

    return mean, log_diag_weights, off_diag_weights, cross_channel_weights

def make_3d_multi_channel_supn_weights(local_connection_dist, im_size_d, im_size_w, im_size_h, num_ch, random_seed, batch_size):
    """
    Create a set of SUPN weights for a multi-channel model
    """
    num_off_diag_weights = get_num_off_diag_weights(local_connection_dist, use_3d=True)
    num_cross_channel_weights = get_num_cross_channel_weights(num_ch)

    torch.manual_seed(random_seed)
    log_diag_weights = Variable(
        0.5 * torch.randn((batch_size, num_ch, im_size_d, im_size_w, im_size_h), dtype=dtype, device=device) - 0.0,
        requires_grad=True)
    mean = Variable(
        0.5 * torch.randn((batch_size, num_ch, im_size_d, im_size_w, im_size_h), dtype=dtype, device=device) - 0.0,
        requires_grad=True)
    off_diag_weights = Variable(
        0.1 * torch.randn((batch_size, num_ch * num_off_diag_weights, im_size_d, im_size_w, im_size_h), dtype=dtype, device=device),
        requires_grad=True)
    
    cross_channel_weights = Variable(
        0.1 * torch.randn((batch_size, num_cross_channel_weights, im_size_d, im_size_w, im_size_h), dtype=dtype, device=device),
        requires_grad=True)

    return mean, log_diag_weights, off_diag_weights, cross_channel_weights


def get_test_data(num_data, num_supns, im_size_w, im_size_h, random_seed, requires_grad, num_ch=1, im_size_d=1):
    torch.manual_seed(random_seed)
    # data = torch.concat((1.0 * torch.randn((batch_size, 1, im_size_w, im_size_h),
    #                                       dtype=dtype, device=device, requires_grad=requires_grad),
    #                     0.5 * torch.randn((batch_size, 1, im_size_w, im_size_h),
    #                                       dtype=dtype, device=device, requires_grad=requires_grad)), dim=1)
    if im_size_d > 1:
        data = torch.randn((num_data,num_supns, num_ch, im_size_d, im_size_w, im_size_h), dtype=dtype, device=device,
                        requires_grad=requires_grad)
    else:
        data = torch.randn((num_data,num_supns, num_ch, im_size_w, im_size_h), dtype=dtype, device=device,
                        requires_grad=requires_grad)

    return data
