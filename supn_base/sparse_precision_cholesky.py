# NDFC Jan 2021 - Toolkit functions to deal with sparse precision in Cholesky form.
#
# NOTES:
#   - At the moment probably want to use the use_transpose=True setting for all the functions - mixing modes
#     is not advised as these are not strict transposes of one-another..
#

import numpy as np
import torch
from torch.autograd import Function
from torch.nn import functional as F
import scipy.sparse as sparse
from typing import Tuple

from supn_base.supn_data import convert_log_to_diag_weights, get_num_off_diag_weights, get_num_cross_channel_weights


def build_off_diag_filters(local_connection_dist, use_transpose=True, device=None, dtype=torch.float, use_3d=False):
    """Create the conv2d/conv3d filter weights for the off-diagonal components of the sparse chol.

    NOTE: Important to specify device if things might run under cuda since constants are created and need to be
        on the correct device.

    Args:
        local_connection_dist (int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose (bool): Defaults to True - usually what we want for the jacobi sampling.
        device: Specify the device to create the constants on (i.e. cpu vs gpu).
        dtype: Specify the dtype to use - defaults to torch.float.
        use_3d (bool): Create 3D filters (i.e. 3x3x3) rather than 2D. Defaults to False.

    Returns:
        tri_off_diag_filters (tensor): [num_off_diag_weights x 1 x [F if use_3d] x F x F] Conv2d/3d kernel filters.
            (Where F = filter_size)
    """
    filter_size = 2 * local_connection_dist + 1
    filter_size_dims_2 = get_num_off_diag_weights(local_connection_dist, use_3d=use_3d)

    if use_transpose:
        tri_off_diag_filters = torch.cat((torch.zeros(filter_size_dims_2, (filter_size_dims_2 + 1),
                                                      device=device, dtype=dtype),
                                          torch.eye(filter_size_dims_2,
                                                    device=device, dtype=dtype)), dim=1)
    else:
        tri_off_diag_filters = torch.cat((torch.fliplr(torch.eye(filter_size_dims_2,
                                                                 device=device, dtype=dtype)),
                                          torch.zeros(filter_size_dims_2, (filter_size_dims_2 + 1),
                                                      device=device, dtype=dtype)), dim=1)

    if use_3d:
        tri_off_diag_filters = torch.reshape(tri_off_diag_filters, (filter_size_dims_2, 1, filter_size, filter_size, filter_size))
    else:
        tri_off_diag_filters = torch.reshape(tri_off_diag_filters, (filter_size_dims_2, 1, filter_size, filter_size))

    return tri_off_diag_filters


def apply_off_diag_weights_offset(off_diag_weights,
                                  local_connection_dist,
                                  use_transpose=True,
                                  reverse_direction=False,
                                  use_3d=False):
    """Shuffle the off-diagonal weights based on the filters to perform the transpose operation.

    Parameters:
        off_diag_weights(tensor): [B ? 1 x F x W x H] off-diagonal terms. F = get_num_off_diag_weights(local_connection_dist)
        local_connection_dist(int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose(bool): Defaults to True.
        reverse_direction(bool): Use the reverse direction for undoing the operation (default False).
        use_3d (bool): Use 3D SUPN model and filters. Defaults to False.

    Returns:
        shuffled_off_diag_weights(tensor): [B ? 1 x F x [D] x W x H].
    """
    if use_3d:
        assert off_diag_weights.ndim == 5
    else:
        assert off_diag_weights.ndim == 4

    tri_off_diag_filters = build_off_diag_filters(local_connection_dist=local_connection_dist,
                                                  use_transpose=not use_transpose,
                                                  use_3d=use_3d)

    channel_filt = torch.split(tri_off_diag_filters, 1, dim=0)

    assert all([torch.nonzero(c).shape[0] == 1 for c in channel_filt])

    if use_3d:
        fD, fW, fH = tri_off_diag_filters.shape[2:]

        im_size_h = off_diag_weights.shape[-1]
        im_size_w = off_diag_weights.shape[-2]

        fD_mid = fD // 2
        fW_mid = fW // 2
        fH_mid = fH // 2

        offset_dim = -4

        def map_idx(idx):
            shuff_idx = - (((idx[0] - fD_mid) * im_size_w * im_size_h) + ((idx[1] - fW_mid) * im_size_h) + (
                        idx[2] - fH_mid))
            if reverse_direction:
                shuff_idx = - shuff_idx
            return shuff_idx

        indices = [map_idx(torch.nonzero(c)[0, 2:]).item() for c in channel_filt]
    else:
        fW, fH = tri_off_diag_filters.shape[2:]

        im_size_h = off_diag_weights.shape[-1]

        fW_mid = fW // 2
        fH_mid = fH // 2

        offset_dim = -3

        def map_idx(idx):
            shuff_idx = - (((idx[0] - fW_mid) * im_size_h) + (idx[1] - fH_mid))
            if reverse_direction:
                shuff_idx = - shuff_idx
            return shuff_idx

        indices = [map_idx(torch.nonzero(c)[0, 2:]).item() for c in channel_filt]

    # if reverse_direction:
    #     assert all([i < 0 for i in indices])
    # else:
    #     assert all([i > 0 for i in indices])

    channel_weights = torch.split(off_diag_weights, 1, dim=offset_dim)
    channel_weights = [torch.roll(cw, offset) for cw, offset in zip(channel_weights, indices)]

    off_diag_weights_shuffled = torch.cat(channel_weights, dim=offset_dim)

    return off_diag_weights_shuffled


def get_prec_chol_as_sparse_tensor(log_diag_weights,
                                   off_diag_weights,
                                   local_connection_dist,
                                   use_transpose=True,
                                   use_3d=False,
                                   cross_ch=None):
    """Returns the precision Cholesky matrix as a sparse COO tensor (on the CPU).

    Args:
        log_diag_weights(tensor): [BATCH x CH x 1 x W x H] log of the diagonal terms (mapped through exp).
        off_diag_weights(tensor): [BATCH x CH x F x W x H] off-diagonal terms.
                                  F = get_num_off_diag_weights(local_connection_dist)
        local_connection_dist (int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose (bool): Defaults to True.
        use_3d (bool): Use 3D SUPN model and filters. Defaults to False.

    Returns:
        sparse_prec_chol (torch.sparse_coo_tensor): The Cholesky factor as a sparse COO precision matrix.
    """
    if use_3d:
        assert log_diag_weights.ndim == 5
        assert off_diag_weights.ndim == 5
    else:
        assert log_diag_weights.ndim == 4
        assert off_diag_weights.ndim == 4

    assert log_diag_weights.device == off_diag_weights.device
    device = off_diag_weights.device
    dtype = off_diag_weights.dtype

    num_ch = log_diag_weights.shape[1]

    # # The sparse tensor package requires the use of doubles..
    # dtype = torch.double

    num_off_diag_weights = get_num_off_diag_weights(local_connection_dist=local_connection_dist,
                                                    use_3d=use_3d)
    tri_off_diag_filters = build_off_diag_filters(local_connection_dist=local_connection_dist,
                                                  use_transpose=use_transpose,
                                                  device=device,
                                                  dtype=dtype,
                                                  use_3d=use_3d)

    #TODO: Check this is OK..
    if not use_transpose:
        off_diag_weights = apply_off_diag_weights_offset(off_diag_weights=off_diag_weights,
                                                         local_connection_dist=local_connection_dist,
                                                         use_transpose=not use_transpose,
                                                         use_3d=use_3d)

    batch_size = log_diag_weights.shape[0]


    if use_3d:
        im_size_D = log_diag_weights.shape[2]
        im_size_H = log_diag_weights.shape[3]
        im_size_W = log_diag_weights.shape[4]

        im_all_size = im_size_D * im_size_H * im_size_W
        view_dims = (1, 1, im_size_D, im_size_H, im_size_W)
        cat_dim = -4
    else:
        im_size_H = log_diag_weights.shape[2]
        im_size_W = log_diag_weights.shape[3]

        im_all_size = im_size_H * im_size_W
        view_dims = (1, 1, im_size_H, im_size_W)
        cat_dim = -3

    diag_values = convert_log_to_diag_weights(log_diag_weights)

    index_input = torch.arange(im_all_size, dtype=dtype, device=device).view(*view_dims) + 1

    # The following is involved and probably not the most efficient way of doing things,
    # it was more focused on being correct (which I think it is!). Essentially we need
    # to determine the batch, row and column indices of the sparse values..

    indices_col = []
    indices_row = []
    values = []

    
    if use_3d:
        off_diag_indices = F.conv3d(index_input.view(-1, 1, im_size_D, im_size_H, im_size_W).double(),
                                    tri_off_diag_filters.double(),
                                    padding=local_connection_dist, stride=1)
    else:
        off_diag_indices = F.conv2d(index_input.view(-1, 1, im_size_H, im_size_W).double(),
                                    tri_off_diag_filters.double(),
                                    padding=local_connection_dist, stride=1)
    # Repeat the process for the multiple channels that we have in the data, adding the channel offset
    # We need to mask the off-diagonal indices to ensure that we don't add extra non-zero values
    off_diag_indices_mask = off_diag_indices > 0
    for ch in range(num_ch):
        ch_offset = ch*(im_all_size)
        indices_col.extend([index_input+ch_offset, (off_diag_indices+ch_offset)*off_diag_indices_mask])
        indices_row.extend((1 + num_off_diag_weights) * [(index_input+ch_offset)])
        values.extend([diag_values[:, ch:ch+1, :, :], off_diag_weights[:, ch*num_off_diag_weights:num_off_diag_weights*(ch+1), :, :]])
    

    # Now need to add the cross-channel weights
    cross_ch_idx = 0
    for ch in range(num_ch-1):
        for ch2 in range(ch+1, num_ch):
            indices_row.append(index_input+ch*im_all_size)
            indices_col.append(index_input+ch2*im_all_size)
            values.append(cross_ch[:, cross_ch_idx:cross_ch_idx+1,...])
            cross_ch_idx += 1


    all_indices_col = torch.cat(indices_col, dim=cat_dim)
    all_indices_row = torch.cat(indices_row, dim=cat_dim)
    all_values = torch.cat(values, dim=cat_dim)

    all_indices_col = all_indices_col.flatten().long()
    all_indices_row = all_indices_row.flatten().long()
    all_values = all_values.flatten()

    all_indices_col_used = all_indices_col[all_indices_col > 0]
    all_indices_row_used = all_indices_row[all_indices_col > 0]
    all_values_used = all_values[all_indices_col.repeat(batch_size).flatten() > 0]

    all_indices_col_used -= 1
    all_indices_row_used -= 1

    all_indices_batch_used = torch.arange(batch_size, device=device).view(-1, 1).expand(batch_size,
                                                                                        all_indices_row_used.shape[0])

    all_indices_batch_used = all_indices_batch_used.flatten()

    all_indices_row_used = all_indices_row_used.repeat(batch_size).flatten()
    all_indices_col_used = all_indices_col_used.repeat(batch_size).flatten()

    sparse_LT_coo = torch.sparse_coo_tensor(indices=torch.stack([all_indices_batch_used, all_indices_row_used, all_indices_col_used]),
                                            values=all_values_used,
                                            size=[batch_size, im_all_size*num_ch, im_all_size*num_ch],
                                            dtype=torch.double)

    return sparse_LT_coo


def apply_sparse_chol_rhs_matmul(dense_input,
                                 log_diag_weights,
                                 off_diag_weights,
                                 local_connection_dist,
                                 use_transpose=True,
                                 use_3d=False,
                                 cross_ch=None):
    """Apply the sparse chol matrix to a dense input on the rhs i.e. result^T = input^T L  (standard matrix mulitply).

    IMPORTANT: Only valid for a single channel at the moment.

    Args:
        dense_input(tensor): [BATCH x 1 x W x H] Input matrix (must be single channel).
        log_diag_weights(tensor): [B ? 1 x 1 x W x H] log of the diagonal terms (mapped through exp).
        off_diag_weights(tensor): [B ? 1 x F x W x H] off-diagonal terms. F = get_num_off_diag_weights(local_connection_dist)
        local_connection_dist(int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose(bool): Defaults to True.
        use_3d (bool): Use 3D SUPN model and filters. Defaults to False.

    Returns:
        product(tensor): [BATCH x 1 x W x H] Result of (L dense_input) or (L^T dense_input).
    """
    if use_3d:
        assert dense_input.ndim == 5
        assert log_diag_weights.ndim == 5
        assert off_diag_weights.ndim == 5
    else:
        assert dense_input.ndim == 4
        assert log_diag_weights.ndim == 4
        assert off_diag_weights.ndim == 4

    # Check how many channels
    num_ch = dense_input.shape[1]

    assert log_diag_weights.dtype == off_diag_weights.dtype
    assert dense_input.dtype == log_diag_weights.dtype

    tri_off_diag_filters = build_off_diag_filters(local_connection_dist=local_connection_dist,
                                                  use_transpose=use_transpose,
                                                  device=dense_input.device,
                                                  dtype=log_diag_weights.dtype,
                                                  use_3d=use_3d)
    
    # need to copy the weights based on the number of channels
    tri_off_diag_filters = torch.cat([tri_off_diag_filters for i in range(num_ch)], dim=0)

    # TODO: Need to resolve the issue around whether or not to use apply_off_diag_weights_offset here for lower mode...
    # assert use_transpose is True

    diag_values = convert_log_to_diag_weights(log_diag_weights)
    
    if use_3d:
        interim = F.conv3d(dense_input, tri_off_diag_filters, padding=local_connection_dist, stride=1, groups=num_ch)
        interim = interim.view(-1, num_ch, interim.shape[1]//num_ch, interim.shape[-3], interim.shape[-2], interim.shape[-1])
        after_weights = torch.einsum('bqfdwh, bqfdwh->bqdwh' if off_diag_weights.shape[0] > 1 else 'bqfdwh, xqfdwh->bqdwh',
                                     interim, off_diag_weights.view((-1,) + interim.shape[1:]))
    else:
        # Use a grouped convolution to apply the replicated off-diag filters separately across channels
        interim = F.conv2d(dense_input, tri_off_diag_filters, padding=local_connection_dist, stride=1, groups=num_ch)
        interim = interim.view(-1, num_ch, interim.shape[1]//num_ch, interim.shape[2], interim.shape[3])

        # Do a channelwise multiply and sum with the off-diagonal weights
        after_weights = torch.einsum('bqfwh, bqfwh->bqwh' if off_diag_weights.shape[0] > 1 else 'bqfwh, xqfwh->bqwh',
                                    interim, off_diag_weights.view((-1,) + interim.shape[1:]))
        
        
    
    result = diag_values * dense_input + after_weights.view(*dense_input.shape)

    # If we have multiple channels and cross channel weights, we need to incorporate these
    if num_ch > 1 and cross_ch is not None:
        # Make our cross channel filters 
        cross_channel_filters = torch.zeros((get_num_cross_channel_weights(num_ch), num_ch, 1, 1), device=dense_input.device, dtype=dense_input.dtype, requires_grad=False)
        if use_3d:
            cross_channel_filters = cross_channel_filters.unsqueeze(-1)
        filter_split_idx = []
        with torch.no_grad():
            # These should be of shape [num_cross_ch x C x 1 x 1], containing a 1 where channel j is connected to channel i (i < j)
            filter_idx = 0
            for i in range(num_ch):
                for j in range(i+1, num_ch):
                    cross_channel_filters[filter_idx, j, ...] = 1
                    filter_idx = filter_idx + 1
                filter_split_idx.append(filter_idx)

        
        # Pick out the right channel information - this is a 1x1 convolution with a single 1 for the correct channel
        if use_3d:
            interim = F.conv3d(dense_input, cross_channel_filters, stride=1)
        else:
            interim = F.conv2d(dense_input, cross_channel_filters, stride=1)

        
        # Multiply elementwise by the cross channel weights
        weighted_interim = interim * cross_ch
        
        # We will have different numbers of entries per channel, so we need to split the result and sum them individually.
        channelwise_interim = torch.tensor_split(weighted_interim, filter_split_idx, dim=1)

        # Sum over the number of filters per channel for each channel, the last should be 0
        channelwise_effect = torch.cat([torch.sum(channelwise_interim[i], dim=1, keepdim=True) for i in range(num_ch)], dim=1)
        
        # Add the result to the previous result
        result = result + channelwise_effect

    

    return result


def log_prob_from_sparse_chol_prec_with_whitened_mean(x,
                                                      whitened_mean,
                                                      log_diag_weights,
                                                      off_diag_weights,
                                                      local_connection_dist,
                                                      use_transpose=True):
    """Calculate the log probability of x under the "whitened mean" and precision Cholesky.

        The whitened mean is defined as (L^T mu) so the real mean needs to be found by
        solving for (L^T)^-1 (L^T mu) = mu.

    Args:
        x(tensor): [BATCH x 1 x W x H] Data, i.e. return log p(x).
        whitened_mean(tensor): [BATCH x 1 x W x H] the whitened mean (L^T mu).
        log_diag_weights(tensor): [B ? 1 x 1 x W x H] log of the diagonal terms (mapped through exp).
        off_diag_weights(tensor): [B ? 1 x F x W x H] off-diagonal terms. F = get_num_off_diag_weights(local_connection_dist)
        local_connection_dist(int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose(bool): Defaults to True.

    Returns:
        log_prob(tensor): [BATCH] The log prob of each element in the batch.
    """
    assert log_diag_weights.ndim == 4
    assert off_diag_weights.ndim == 4

    assert log_diag_weights.shape[1] == 1
    im_size_w = log_diag_weights.shape[2]
    im_size_h = log_diag_weights.shape[3]

    fitting_term = apply_sparse_chol_rhs_matmul(x,
                                                log_diag_weights=log_diag_weights,
                                                off_diag_weights=off_diag_weights,
                                                local_connection_dist=local_connection_dist,
                                                use_transpose=use_transpose)

    fitting_term = fitting_term - whitened_mean

    constant_term = im_size_w * im_size_h * torch.log(torch.Tensor([2.0]) * np.pi)
    constant_term = constant_term.to(log_diag_weights.device)

    # Can't do this in case we do something funning to the diag weights..
    log_det_term = 2.0 * torch.sum(log_diag_weights, dim=(1,2,3,)) # Note these are precision NOT covariance L

    # Do this in case we do something funny in the conversion..
    actual_log_diag_values = torch.log(convert_log_to_diag_weights(log_diag_weights))
    log_det_term = 2.0 * torch.sum(actual_log_diag_values, dim=(1, 2, 3,))  # Note these are precision NOT covariance L

    log_prob = -0.5 * torch.sum(torch.square(fitting_term), dim=(1,2,3,)) \
               -0.5 * constant_term \
               +0.5 * log_det_term # Note positive since precision..

    return log_prob


def log_prob_from_sparse_chol_prec(x,
                                   mean,
                                   log_diag_weights,
                                   off_diag_weights,
                                   local_connection_dist,
                                   use_transpose=True,
                                   use_3d=False,
                                   cross_ch=None):
    """Calculate the log probability of x under the mean and precision Cholesky.

    Args:
        x(tensor): [BATCH x 1 x W x H] Data, i.e. return log p(x).
        mean(tensor): [BATCH x 1 x W x H] the mean (mu).
        log_diag_weights(tensor): [B ? 1 x 1 x W x H] log of the diagonal terms (mapped through exp).
        off_diag_weights(tensor): [B ? 1 x F x W x H] off-diagonal terms. F = get_num_off_diag_weights(local_connection_dist)
        local_connection_dist(int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose(bool): Defaults to True.
        use_3d (bool): Use 3D SUPN model and filters. Defaults to False.

    Returns:
        log_prob(tensor): [BATCH] The log prob of each element in the batch.
    """
    if use_3d:
        assert x.ndim == 5
        assert mean.ndim == 5
        assert log_diag_weights.ndim == 5
        assert off_diag_weights.ndim == 5

        assert x.shape == mean.shape
        assert log_diag_weights.shape[2:] == x.shape[2:]

        im_size_d = log_diag_weights.shape[2]
        im_size_w = log_diag_weights.shape[3]
        im_size_h = log_diag_weights.shape[4]

        all_size = im_size_d * im_size_w * im_size_h
        dims_to_sum = (1, 2, 3, 4,)
    else:
        assert log_diag_weights.ndim == 4
        assert off_diag_weights.ndim == 4

        im_size_w = log_diag_weights.shape[2]
        im_size_h = log_diag_weights.shape[3]

        all_size = im_size_w * im_size_h
        dims_to_sum = (1, 2, 3,)

    assert use_transpose
    fitting_term = apply_sparse_chol_rhs_matmul(x - mean,
                                                log_diag_weights=log_diag_weights,
                                                off_diag_weights=off_diag_weights,
                                                local_connection_dist=local_connection_dist,
                                                use_transpose=use_transpose,
                                                use_3d=use_3d,
                                                cross_ch=cross_ch)

    constant_term = all_size * torch.log(torch.Tensor([2.0]) * np.pi)
    constant_term = constant_term.to(log_diag_weights.device)

    # Can't do this in case we do something funny to the diag weights...
    # log_det_term = 2.0 * torch.sum(log_diag_weights, dim=dims_to_sum) # Note these are precision NOT covariance L

    # Do this in case we do something funny in the conversion...
    actual_log_diag_values = torch.log(convert_log_to_diag_weights(log_diag_weights))
    log_det_term = 2.0 * torch.sum(actual_log_diag_values, dim=dims_to_sum)  # Note these are precision NOT covariance L

    log_prob = -0.5 * torch.sum(torch.square(fitting_term), dim=dims_to_sum) \
               -0.5 * constant_term \
               +0.5 * log_det_term # Note positive since precision..

    return log_prob

def coo_to_supn_sparse(coo_prec_chol: torch.sparse_coo_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Method to convert a sparse tensor in coo format to a supn formatted
    precision cholesky. Tested on square images for single distribution supn models
    Assumes the input is a square matrix with a single channel

    Parameters
    ----------
    coo_prec_chol : torch.sparse.COO
    sparse tensor in coo format
    Returns
    -------
    log_diag_weights : torch.Tensor
    log diagonal weights in supn format
    off_diag_weights : torch.Tensor
    off diagonal weights in supn format
    '''
    im_w = int(torch.sqrt(torch.tensor(coo_prec_chol.shape[1])))
    im_h = im_w

    diagonals = []
    for i in range(coo_prec_chol.shape[1]):
        if (coo_prec_chol[0].to_dense().diagonal(offset=i)**2).sum() != 0:
            diagonals.append(torch.cat([coo_prec_chol[0].to_dense().diagonal(offset=i),torch.zeros(i,device='cuda')]))
    supn_chol = torch.stack(diagonals)
    log_diag_weights = torch.log(supn_chol[0])
    off_diag_weights = supn_chol[1:]
    return log_diag_weights.reshape([1,1,im_w,im_h]), off_diag_weights.reshape([1,-1,im_w,im_h])
