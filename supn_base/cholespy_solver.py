from enum import Enum

import numpy as np
import torch
from torch.autograd import Function
from torch.nn import functional as F
from supn_base.supn_data import SUPNData, convert_log_to_diag_weights, get_num_off_diag_weights

from cholespy import SupnCholeskySolverF, SupnCholeskySolverD, MatrixType, inspect

from supn_base.sparse_precision_cholesky import build_off_diag_filters, apply_off_diag_weights_offset, get_prec_chol_as_sparse_tensor
# import torch_sparse_solve as tss
# import scipy.sparse as sparse


def extract_correct_off_diag_entries(dense_input, local_connection_dist, use_transpose=True,
                                     dtype=torch.float, use_3d=False):
    """Extracts the off-diagonal entries correctly to build up the gradients.

    IMPORTANT: Only valid for a single channel at the moment.

    Parameters:
        dense_input(tensor): [BATCH x SAMPLES x CHANNELS x [D] x W x H] Input matrix (must be single channel).
        local_connection_dist(int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose(bool): Defaults to True.
        dtype: Specify the dtype to use - defaults to torch.float.
        use_3d (bool): Use 3D SUPN model and filters. Defaults to False.

    Returns:
        off-diagonal entries(tensor): [BATCH x F x [D] x W x H]
            Where F = get_num_off_diag_weights(local_connection_dist, use_3d).
    """
    if use_3d:
        assert dense_input.ndim == 6
        num_ch = dense_input.shape[-4]
    else:
        assert dense_input.ndim == 5
        num_ch = dense_input.shape[-3]

    tri_off_diag_filters = build_off_diag_filters(local_connection_dist=local_connection_dist,
                                                  use_transpose=use_transpose,
                                                  dtype=dense_input.dtype,
                                                  device=dense_input.device,
                                                  use_3d=use_3d)
    
    if use_3d:
        im_size_d = dense_input.shape[-3]
        im_size_w = dense_input.shape[-2]
        im_size_h = dense_input.shape[-1]
        extracted = F.conv3d(dense_input.reshape(-1, num_ch, im_size_d, im_size_w, im_size_h),
                             tri_off_diag_filters, padding=local_connection_dist, stride=1)
    else:
        im_size_w = dense_input.shape[-2]
        im_size_h = dense_input.shape[-1]

        extracted = F.conv2d(dense_input.reshape(-1, num_ch, im_size_w, im_size_h),
                             tri_off_diag_filters, padding=local_connection_dist, stride=1)

    # return extracted.view(*dense_input.shape[1:-2], *extracted.shape[1:])
    # Returns [num_batches x num_samples x num_off_diag x [D] x W x H]
    return extracted.view(*dense_input.shape[0:2], *extracted.shape[1:])


def get_sparse_LT_matrix_index_values(supn_data: SUPNData, use_transpose=True):
    """
        This function is not actually used during the solve, but instead to calculate the indices of the sparse matrix
       Parameters:
        supn_data(SUPNData): The SUPN data.
        use_transpose (bool): Defaults to True.
    :return: the rows, cols, values, and overall shape of the sparse matrix.
    """
    log_diag = supn_data.log_diag[0].unsqueeze(0)
    off_diag = supn_data.off_diag[0].unsqueeze(0)
    local_connection_dist = supn_data.local_connection_dist
    cross_ch = supn_data.cross_ch
    use_3d = supn_data.use_3d
    
    num_ch = supn_data.get_num_ch()
    num_F = get_num_off_diag_weights(local_connection_dist=local_connection_dist, use_3d=use_3d)
    supn_data.test_consistency()

    # This is not checked for anything else - will probably mess up since would need to shuffle off_diags...
    assert use_transpose is True

    # If we have multiple channels, we only use the first one for calculating the indicies - and can then add an offset for subsequent channels
    if num_ch > 1:
        log_diag = log_diag[:, 0:1,...]
        off_diag = off_diag[:, 0:num_F, ...]  

    # We can only do batch size 1 at the moment...
    assert log_diag.shape[0] == 1
    assert off_diag.shape[0] == 1

    assert log_diag.device == off_diag.device
    device = off_diag.device
    dtype = off_diag.dtype

    num_off_diag_weights = get_num_off_diag_weights(local_connection_dist=local_connection_dist,
                                                    use_3d=use_3d)
    tri_off_diag_filters = build_off_diag_filters(local_connection_dist=local_connection_dist,
                                                  use_transpose=use_transpose,
                                                  device=device,
                                                  dtype=dtype,
                                                  use_3d=use_3d)

    batch_size = log_diag.shape[0]

    if use_3d:
        im_size_D = log_diag.shape[2]
        im_size_H = log_diag.shape[3]
        im_size_W = log_diag.shape[4]

        im_all_size = im_size_D * im_size_H * im_size_W
        view_dims = (1, 1, im_size_D, im_size_H, im_size_W)
        cat_dim = -4
    else:
        im_size_H = log_diag.shape[2]
        im_size_W = log_diag.shape[3]

        im_all_size = im_size_H * im_size_W
        view_dims = (1, 1, im_size_H, im_size_W)
        cat_dim = -3

    assert batch_size == 1

    diag_values = convert_log_to_diag_weights(log_diag)

    diag_values = torch.reshape((torch.arange(diag_values.numel(), dtype=dtype, device=device) + 1.0),
                                diag_values.shape)
    off_diag = torch.reshape(torch.arange(off_diag.numel(), dtype=dtype, device=device) +
                                     diag_values.numel() + 1.0, off_diag.shape)

    index_input = torch.arange(im_all_size, device=device).view(*view_dims) + 1

    if use_3d:
        off_diag_indices = F.conv3d(index_input.view(-1, 1, im_size_D, im_size_H, im_size_W).double(),
                                    tri_off_diag_filters.double(),
                                    padding=local_connection_dist, stride=1)
    else:
        off_diag_indices = F.conv2d(index_input.view(-1, 1, im_size_H, im_size_W).double(),
                                    tri_off_diag_filters.double(),
                                    padding=local_connection_dist, stride=1)

     # Get the column indicies for this entry
    all_indices_col = torch.cat((index_input, off_diag_indices), dim=cat_dim)

    
    # Add a mask for the zeros to avoid adding extra elements when stacking channels
    all_indices_col_mask = (all_indices_col > 0).float()
    # We need to stack the different channels (if there are any) adding a suitable offset while masking the zeros
    all_indices_col = torch.cat([(all_indices_col + (im_all_size * ch))*all_indices_col_mask for ch in range(num_ch)], dim=cat_dim)
    
    # Get the row indices for this entry (replicating the index_input)
    all_indices_row = torch.cat((1 + num_off_diag_weights) * [index_input], dim=cat_dim)
    # Mask out the zeros
    all_indices_row_mask = (all_indices_row > 0).float()
    
    # Again, stack the different channels (if there are any) adding a suitable offset while masking the zeros
    all_indices_row = torch.cat([(all_indices_row + im_all_size * ch) * all_indices_row_mask for ch in range(num_ch)], dim=cat_dim)
    
    
    if num_ch > 1:
        all_values = []
        # Calculate the number of pixels and filters (i.e. elements per channel)
        num_pix_chs = im_all_size * (num_F+1)
        for i in range(num_ch):
            index_offset = (num_pix_chs*i)
            # Stack together the diagonal and off-diagonal vector indices
            all_values.append(torch.cat((diag_values + index_offset, off_diag + index_offset), dim=cat_dim))
        
        all_values = torch.cat(all_values, dim=cat_dim)

        
        # The cross channel weights come last - make a list of indicies for each cross-channel image
        cross_ch_weights_offset = num_pix_chs*num_ch
        cross_indices_col = [all_indices_col]
        cross_indices_row = [all_indices_row]
        
        cross_ch_image_idx = 0
        index_input_mask = (index_input > 0).float()
        # Add the off-diagonal cross-channel elements
        for i in range(num_ch):
            for j in range(i+1, num_ch):
                cross_indices_row.append((index_input + im_all_size * i)*index_input_mask) 
                cross_indices_col.append((index_input + im_all_size * j)*index_input_mask)
                # Make sure we point to the correct raw input value
                all_values = torch.cat([all_values, diag_values+cross_ch_weights_offset+cross_ch_image_idx*im_all_size], dim=cat_dim)
                cross_ch_image_idx += 1
        
        all_indices_col = torch.cat(cross_indices_col, dim=cat_dim)
        all_indices_row = torch.cat(cross_indices_row, dim=cat_dim)

    else:
        # Stack together the diagonal and off-diagonal vector indices
        all_values = torch.cat((diag_values, off_diag), dim=cat_dim)

    all_indices_col = all_indices_col.flatten().long()
    all_indices_row = all_indices_row.flatten().long() 
    all_values = all_values.flatten()

    all_indices_col_used = all_indices_col[all_indices_col > 0]
    all_indices_row_used = all_indices_row[all_indices_col > 0]
    all_values_used = all_values[all_indices_col > 0]

    all_indices_col_used -= 1
    all_indices_row_used -= 1

    rows = all_indices_row_used
    cols = all_indices_col_used
    values = all_values_used
    shape = [num_ch * im_all_size, num_ch*im_all_size]

    return rows, cols, values, shape


class SupnSolveType(Enum):
    LOWER = 1,
    UPPER = 2,
    PRECISION = 3,


class SUPNSolver:
    def __init__(self, supn_data: SUPNData):
        assert isinstance(supn_data, SUPNData)
        log_diag = supn_data.log_diag
        off_diag = supn_data.off_diag
        cross_ch = supn_data.cross_ch
        self._use_3d = supn_data.use_3d
        self._local_connection_dist = supn_data.local_connection_dist
         

        # Calculate the size length of the matrix as number of pixels/voxels X channels
        if self._use_3d:
            assert log_diag.ndim == 5
            assert off_diag.ndim == 5
            self._n_elems = log_diag.shape[-1] * log_diag.shape[-2] * log_diag.shape[-3] * log_diag.shape[-4]
        else:
            assert log_diag.ndim == 4
            assert off_diag.ndim == 4
            self._n_elems = log_diag.shape[-1] * log_diag.shape[-2] * log_diag.shape[-3]

        # Just for building the right datastructures - let's only use a single batch.
        # Can still use mutliple batches when running the solve...
        if log_diag.shape[0] > 1:
            log_diag = log_diag[0:1]
        if off_diag.shape[0] > 1:
            off_diag = off_diag[0:1]
        if cross_ch is not None:
            cross_ch= cross_ch[0:1]

        supn_data.test_consistency()

        rows, cols, index_values, shape = get_sparse_LT_matrix_index_values(supn_data, use_transpose=True,)
        self._device = log_diag.device
        assert log_diag.device == off_diag.device
        assert log_diag.dtype == off_diag.dtype

        raw_values = self._get_raw_values(log_diag, off_diag, cross_ch)

        if self._use_3d:
            assert raw_values.ndim == 5
        else:
            assert raw_values.ndim == 4
        assert raw_values.shape[0] == 1

        raw_values = raw_values.flatten()

        sparse_LT_coo = torch.sparse_coo_tensor(indices=torch.stack([rows, cols]),
                                                values=index_values,
                                                size=shape,
                                                dtype=torch.float)
        
        # If we ever want to check the sparsity patterns match the other approach, uncomment this
        if False:
            dense_LT_coo = sparse_LT_coo.to_dense()
            dense_LT_v2 = get_prec_chol_as_sparse_tensor(log_diag_weights=log_diag, off_diag_weights=off_diag, local_connection_dist=supn_data.local_connection_dist, use_transpose=True, use_3d=self._use_3d, cross_ch=supn_data.cross_ch).to_dense()
            # Check the sparsity pattern matches the other approach... the values won't match as this COO has indicies of values rather than actual values
            assert(torch.allclose((dense_LT_v2.abs()>0).float(),(dense_LT_coo.abs()>0).float()))


        assert shape[1] == shape[0]

        sparse_LT_csr = sparse_LT_coo.to_sparse_csr()

        upper_csr_rows = sparse_LT_csr.crow_indices().int()
        upper_csr_cols = sparse_LT_csr.col_indices().int()
        upper_csr_index_data = sparse_LT_csr.values()

        sparse_L_csr = torch.transpose(sparse_LT_coo, 0, 1).to_sparse_csr()

        lower_csr_rows = sparse_L_csr.crow_indices().int()
        lower_csr_cols = sparse_L_csr.col_indices().int()
        lower_csr_index_data = sparse_L_csr.values()

        csr_n_rows = sparse_L_csr.shape[0]

        # Remember to take 1 off to go back to zero based index!!
        upper_csr_indices = upper_csr_index_data.int() - 1
        lower_csr_indices = lower_csr_index_data.int() - 1

        if log_diag.dtype == torch.float:
            self._supn_chol_solver = SupnCholeskySolverF(csr_n_rows,
                                                         lower_csr_rows, lower_csr_cols, lower_csr_indices,
                                                         upper_csr_rows, upper_csr_cols, upper_csr_indices,
                                                         raw_values)
        elif log_diag.dtype == torch.float64:
            self._supn_chol_solver = SupnCholeskySolverD(csr_n_rows,
                                                         lower_csr_rows, lower_csr_cols, lower_csr_indices,
                                                         upper_csr_rows, upper_csr_cols, upper_csr_indices,
                                                         raw_values)
        else:
            assert False

    @property
    def use_3d(self):
        return self._use_3d

    def _get_raw_values(self, log_diag_weights, off_diag_weights, cross_ch):
        # return torch.concat([torch.exp(log_diag_weights) + MIN_DIAG_VALUE,
        #                            off_diag_weights], dim=1)
        if log_diag_weights.shape[1] == 1:
            diag_values = convert_log_to_diag_weights(log_diag_weights)
            return torch.concat([diag_values, off_diag_weights], dim=1)
        else:
            num_ch = log_diag_weights.shape[1]
            diag_weights_list = torch.split(convert_log_to_diag_weights(log_diag_weights), 1, dim=1)
            off_diag_weights_list = torch.split(off_diag_weights, get_num_off_diag_weights(self._local_connection_dist, self._use_3d), dim=1)
            to_concat = []
            for ch in range(num_ch):
                to_concat.append(diag_weights_list[ch])
                to_concat.append(off_diag_weights_list[ch])

            if cross_ch is not None:
                to_concat.append(cross_ch)

            return torch.concat(to_concat, dim=1)

    def test_sparse_matrices(self, log_diag_weights, off_diag_weights):
        raw_values = self._get_raw_values(log_diag_weights, off_diag_weights)
        raw_values = raw_values.flatten()

        assert log_diag_weights.dtype == off_diag_weights.dtype

        def get_matrix(lower):
            rr = torch.ones(self._supn_chol_solver.get_n_rows() + 1, dtype=torch.int32, device=self._device)
            cc = torch.ones(self._supn_chol_solver.get_n_entries(), dtype=torch.int32, device=self._device)
            dd = torch.ones(self._supn_chol_solver.get_n_entries(), dtype=torch.int32, device=self._device)
            vv = torch.ones(self._supn_chol_solver.get_n_raw_data(), dtype=log_diag_weights.dtype, device=self._device)

            self._supn_chol_solver.debug_print(rr, cc, dd, vv, lower)

            return torch.sparse_csr_tensor(rr, cc, raw_values[dd.long()], size=[self._n_elems, self._n_elems], dtype=log_diag_weights.dtype)

        LL = get_matrix(lower=True)
        # print('LL', LL, '\n', LL.to_dense())

        LT = get_matrix(lower=False)
        # print('LT', LT, '\n', LT.to_dense())

        return LL, LT

    def solve_with_lower_no_grads(self, log_diag, off_diag, cross_ch, rhs):
        """
        Solve with the lower triangular matrix, but without any gradient computation. To calculate gradients call supn_lower_cholesky_solve
        """

        supn_data = SUPNData(mean=log_diag, log_diag=log_diag, off_diag=off_diag, cross_ch=cross_ch, local_connection_dist=self._local_connection_dist, use_3d=self._use_3d)
        return self._solve(supn_data, rhs, skip_lower=False, skip_upper=True)

    def solve_with_upper_no_grads(self, log_diag, off_diag, cross_ch, rhs):
        """
        Solve with the upper triangular matrix, but without any gradient computation. To calculate gradients call supn_upper_cholesky_solve
        """
        supn_data = SUPNData(mean=log_diag, log_diag=log_diag, off_diag=off_diag, cross_ch=cross_ch, local_connection_dist=self._local_connection_dist, use_3d=self._use_3d)
        return self._solve(supn_data, rhs, skip_lower=True, skip_upper=False)

    def solve_with_precision_no_grads(self,  log_diag, off_diag, cross_ch, rhs):
        """
        Solve with the precision matrix, but without any gradient computation. To calculate gradients call supn_precision_cholesky_solve
        """
        supn_data = SUPNData(mean=log_diag, log_diag=log_diag, off_diag=off_diag, cross_ch=cross_ch, local_connection_dist=self._local_connection_dist, use_3d=self._use_3d)
        return self._solve(supn_data, rhs, skip_lower=False, skip_upper=False)

    def _solve(self, supn_data: SUPNData, rhs, skip_lower, skip_upper):
        # We expect the right hand side to be of shape [BATCH x SAMPLES x CHANNEL x W x H]
        if supn_data.use_3d:
            assert rhs.ndim == 6
        else:
            assert rhs.ndim == 5

        log_diag = supn_data.log_diag
        off_diag = supn_data.off_diag

        # Check batch sizes match...
        assert rhs.shape[0] == log_diag.shape[0]
        assert off_diag.shape[0] == log_diag.shape[0]

        # Batch size of 1 at the moment...
        # assert rhs.shape[0] == 1

        batch_shape = rhs.shape[0]

        # assert rhs.shape[1] == 1
        if supn_data.use_3d:
            assert rhs.shape[-4] * rhs.shape[-3] * rhs.shape[-2] * rhs.shape[-1] == self._supn_chol_solver.get_n_rows()
        else:
            assert rhs.shape[-3] * rhs.shape[-2] * rhs.shape[-1] == self._supn_chol_solver.get_n_rows()

        assert not (skip_lower and skip_upper)

        rhs_shape = rhs.shape

        # The [1] index is the number of samples to draw - move it to the end
        if rhs_shape[1] > 1:
            rhs = torch.transpose(rhs.reshape(batch_shape, -1, self._n_elems), 1, 2)
            rhs = rhs.reshape(batch_shape, self._n_elems, -1)
        else:
            rhs = rhs.view(batch_shape, self._supn_chol_solver.get_n_rows(), 1)

        
        assert rhs.shape[1] == self._supn_chol_solver.get_n_rows()
        rhs = rhs.contiguous()

        raw_values = self._get_raw_values(log_diag, off_diag, supn_data.cross_ch)
        
        # raw_values = raw_values.flatten()
        raw_values = raw_values.view(batch_shape, -1)

        raw_values = raw_values.contiguous()

        assert raw_values.shape[1] == self._supn_chol_solver.get_n_raw_data()

        solution = torch.zeros_like(rhs)

        # print(f'self._supn_chol_solver.solve:\n  raw_values={raw_values.shape}\n  rhs={rhs.shape}\n  solution={solution.shape}\n')

        self._supn_chol_solver.solve(raw_values, rhs, solution, skip_lower, skip_upper)

        solution = torch.transpose(solution, 1, 2)

        solution = solution.view(*rhs_shape)

        return solution


def supn_upper_cholesky_solve(supn_solver: SUPNSolver,
                              log_diag_weights,
                              off_diag_weights,
                              cross_ch,
                              rhs):
    '''Applies L^-T @ rhs for the specified log_diag_weights and off_diag_weights with a compatible supn_solver object.

    Args:
        supn_solver: SUPNSolver with compatible log_diag_weights and off_diag_weights of matching dtype and device.
        log_diag_weights(tensor): [1 x 1 x W x H] log of the diagonal terms (mapped through exp).
        off_diag_weights(tensor): [1 x F x W x H] off-diagonal terms. F = get_num_off_diag_weights(local_connection_dist).
        rhs(tensor): [1 x N x W x H] right hand side to solve for (can solve N < 128 in parallel for same L).

    Returns(tensor): [1 x N x W x H] result of L^-T @ rhs.

    '''
    return SUPNSolverCholeskyFunction.apply(supn_solver,
                                            log_diag_weights,
                                            off_diag_weights,
                                            cross_ch,
                                            rhs,
                                            SupnSolveType.UPPER)


def supn_lower_cholesky_solve(supn_solver: SUPNSolver,
                              log_diag_weights,
                              off_diag_weights,
                              cross_ch,
                              rhs):
    '''Applies L^-1 @ rhs for the specified log_diag_weights and off_diag_weights with a compatible supn_solver object.

    Args:
        supn_solver: SUPNSolver with compatible log_diag_weights and off_diag_weights of matching dtype and device.
        log_diag_weights(tensor): [1 x 1 x W x H] log of the diagonal terms (mapped through exp).
        off_diag_weights(tensor): [1 x F x W x H] off-diagonal terms. F = get_num_off_diag_weights(local_connection_dist).
        rhs(tensor): [1 x N x W x H] right hand side to solve for (can solve N < 128 in parallel for same L).

    Returns(tensor): [1 x N x W x H] result of L^-1 @ rhs.

    '''
    return SUPNSolverCholeskyFunction.apply(supn_solver,
                                            log_diag_weights,
                                            off_diag_weights,
                                            cross_ch,
                                            rhs,
                                            SupnSolveType.LOWER)


def supn_precision_solve(supn_solver: SUPNSolver,
                         log_diag_weights,
                         off_diag_weights,
                         cross_ch,
                         rhs):
    '''Applies (L L^T)^-1 @ rhs for the specified log_diag_weights and off_diag_weights with a compatible supn_solver object.

    Args:
        supn_solver: SUPNSolver with compatible log_diag_weights and off_diag_weights of matching dtype and device.
        log_diag_weights(tensor): [1 x 1 x W x H] log of the diagonal terms (mapped through exp).
        off_diag_weights(tensor): [1 x F x W x H] off-diagonal terms. F = get_num_off_diag_weights(local_connection_dist).
        rhs(tensor): [1 x N x W x H] right hand side to solve for (can solve N < 128 in parallel for same L).

    Returns(tensor): [1 x N x W x H] result of (L L^T)^-1 @ rhs (i.e. a solve with the precision matrix).

    '''

    # FOR THE TIME BEING WE DO THE INEFFICIENT SOLVE..

    interim = supn_lower_cholesky_solve(supn_solver, log_diag_weights, off_diag_weights, cross_ch, rhs)
    return supn_upper_cholesky_solve(supn_solver, log_diag_weights, off_diag_weights, cross_ch, interim)

    # return SUPNSolverCholeskyFunction.apply(supn_solver,
    #                                         log_diag_weights,
    #                                         off_diag_weights,
    #                                         rhs,
    #                                         SupnSolveType.PRECISION)


class SUPNSolverCholeskyFunction(Function):
    @staticmethod
    def forward(ctx,
                supn_solver: SUPNSolver,
                log_diag_weights,
                off_diag_weights,
                cross_ch,
                rhs,
                solve_type: SupnSolveType):
        # Call the solver function - we don't need to keep
        # gradients through the forward pass..
        with torch.no_grad():
            if solve_type == SupnSolveType.LOWER:
                output = supn_solver.solve_with_lower_no_grads(log_diag_weights, off_diag_weights, cross_ch, rhs)
            elif solve_type == SupnSolveType.UPPER:
                output = supn_solver.solve_with_upper_no_grads(log_diag_weights, off_diag_weights, cross_ch, rhs)
            elif solve_type == SupnSolveType.PRECISION:
                output = supn_solver.solve_with_precision_no_grads(log_diag_weights, off_diag_weights, cross_ch, rhs)

        # These are the only things we need to save (don't need interim things)..
        ctx.save_for_backward(log_diag_weights, off_diag_weights, cross_ch, rhs, output)

        ctx.extra_data = supn_solver, solve_type
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Unpack saved tensors..
        log_diag_weights, off_diag_weights, cross_ch, rhs, output = ctx.saved_tensors
        supn_solver, solve_type = ctx.extra_data
        assert isinstance(supn_solver, SUPNSolver)

        num_ch = log_diag_weights.shape[1]

        # Init the grads to None in case not required..
        grad_log_diag_weights = grad_off_diag_weights = grad_cross_ch = None

        # Inputs without gradients still need to return None..
        grad_supn_solver = grad_rhs = grad_lower = None

        # Calc if either grad needed..
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2] or ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:

            local_connection_dist = supn_solver._local_connection_dist
            if solve_type == SupnSolveType.LOWER:
                b = supn_solver.solve_with_upper_no_grads(log_diag_weights, off_diag_weights, cross_ch, grad_output)
                use_transpose_for_extract = False
            elif solve_type == SupnSolveType.UPPER:
                b = supn_solver.solve_with_lower_no_grads(log_diag_weights, off_diag_weights, cross_ch, grad_output)
                use_transpose_for_extract = True
            elif solve_type == SupnSolveType.PRECISION:
                # b = supn_solver.solve_with_precision(log_diag_weights, off_diag_weights, grad_output)

                b_tmp = supn_solver.solve_with_lower_no_grads(log_diag_weights, off_diag_weights, cross_ch, grad_output)
                b = supn_solver.solve_with_upper_no_grads(log_diag_weights, off_diag_weights, cross_ch, b_tmp)

                use_transpose_for_extract = False
                print('^^^^^ BLASH BASLSDLFSLDKJGLKDSJFGHSLKDF')

            # CHECK THIS..
            grad_rhs = b

            if solve_type == SupnSolveType.PRECISION:
                # Need to square the diags for the precision (or mult log by 2)
                grad_log_diag_weights = - torch.sum(output * b * torch.exp(2.0 * log_diag_weights),
                                                    dim=1, keepdim=True)
            else:
                # Inset an extra channel for summing over samples - it worked okay when channels was 1 but for multiple channels it's not so good.
                grad_log_diag_weights = - torch.sum(output * b * torch.exp(torch.unsqueeze(log_diag_weights,1)), dim=1)
                #grad_log_diag_weights = - torch.sum(output * b * torch.exp(log_diag_weights), dim=1)
            
            # Cycle over chanels and extract the off-diagonal gradients individually
            grad_off_diag_weights = []
            for ch in range(num_ch):
                dense_ch_input = b[:,:, ch:ch+1, ...]
                output_ch = output[:,:, ch:ch+1, ...]
                # Important that use_transpose is opposite for this one..
                mod_off_diag_weights = extract_correct_off_diag_entries(dense_input=dense_ch_input,
                                                                        local_connection_dist=local_connection_dist,
                                                                        use_transpose=not use_transpose_for_extract,
                                                                        dtype=off_diag_weights.dtype,
                                                                        use_3d=supn_solver.use_3d)
                # Returns [num_batches x num_samples x num_off_diag x W x H]
                grad_off_diag_weights_shuffled = - torch.sum(output_ch *
                                                            mod_off_diag_weights,
                                                            dim=1)

                # Important that reverse_direction is True for this one..
                if solve_type == SupnSolveType.LOWER:
                    grad_off_diag_weights_ch = grad_off_diag_weights_shuffled
                elif solve_type == SupnSolveType.UPPER:
                    grad_off_diag_weights_ch = apply_off_diag_weights_offset(off_diag_weights=grad_off_diag_weights_shuffled,
                                                                        local_connection_dist=local_connection_dist,
                                                                        use_transpose=use_transpose_for_extract,
                                                                        reverse_direction=True,
                                                                        use_3d=supn_solver.use_3d)
                grad_off_diag_weights.append(grad_off_diag_weights_ch)

            grad_off_diag_weights = torch.cat(grad_off_diag_weights, dim=1)

            # Now tackle the cross channel weights
            if num_ch > 1:
                grad_cross_ch = []
                if use_transpose_for_extract:
                    lhs = b
                    rhs = output
                else:
                    lhs = output
                    rhs = b

                for i in range(num_ch):
                    for j in range(i+1, num_ch):
                        grad_cross_ch.append(-torch.sum(lhs[:,:, i:i+1, ...] * rhs[:,:, j:j+1, ...], dim=1))
                grad_cross_ch = torch.cat(grad_cross_ch, dim=1)

    
        return grad_supn_solver, grad_log_diag_weights, grad_off_diag_weights, grad_cross_ch, grad_rhs, grad_lower


def sparse_chol_linsolve(supn_data: SUPNData,
                         rhs: torch.Tensor,
                         supn_solver:SUPNSolver = None, solve_with_upper=True) -> torch.Tensor:
    """
    Solve a linear system with the sparse precision matrix.
    Args:
        log_diag (torch.Tensor): Log diagonal of the precision
        off_diag (torch.Tensor): Cholesky terms of the precision
        rhs (torch.Tensor): Right hand side of the linear system.
        solve_with_upper (bool): If True, solve with the upper triangular matrix. Otherwise, solve with the lower triangular matrix. 
        If we are sampling, then we want to solve with upper! x = (L^T)^-1 z, where L^T is the upper triangular matrix.
    Returns:
        x (torch.Tensor): Solution to the linear system.
    """
    if supn_solver is None:
        supn_solver = SUPNSolver(supn_data)
    if solve_with_upper:
        x = supn_upper_cholesky_solve(supn_solver, supn_data.log_diag, supn_data.off_diag, supn_data.cross_ch, rhs)
    else:
        x = supn_lower_cholesky_solve(supn_solver, supn_data.log_diag, supn_data.off_diag, supn_data.cross_ch, rhs)
    return x

def sample_zero_mean(supn_data: SUPNData, 
                     num_samples: int,
                     supn_solver: SUPNSolver = None) -> torch.Tensor:
    """
    Sample correlated noise from the given log variance and Cholesky terms.
    Args
    x_logvar (torch.Tensor): Log diagonal of the precision
    x_cholesky_terms (torch.Tensor): Cholesky terms of the precision
    Returns:
    z torch.Tensor: Sampled correlated noise.
    """
    original_shape = supn_data.log_diag.shape
    noise_white = torch.randn((supn_data.log_diag.shape[0], num_samples,)+supn_data.log_diag.shape[1:], device=supn_data.log_diag.device, dtype=supn_data.log_diag.dtype, requires_grad=False)
    noise_supn_correlated = sparse_chol_linsolve(supn_data, noise_white, supn_solver, solve_with_upper=True)
    noise_supn_correlated = noise_supn_correlated.permute([1,0,2,3,4])
    return noise_supn_correlated
