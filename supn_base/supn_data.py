from dataclasses import dataclass
import torch 

#initial data class to hold all of the supn dist parameters. 

@dataclass
class Parent:
    def __post_init__(self):
        for (name, field_type) in self.__annotations__.items():
            if not isinstance(self.__dict__[name], field_type):
                current_type = type(self.__dict__[name])
                raise TypeError(f"The field `{name}` was assigned by `{current_type}` instead of `{field_type}`")


@dataclass
class SUPNData(Parent):
    mean: torch.Tensor
    log_diag: torch.Tensor
    off_diag: torch.Tensor
    cross_ch: torch.Tensor = None
    local_connection_dist: int = 2
    use_3d: bool = False

    def __post_init__(self):
        if self.use_3d:
            assert self.mean.ndim == 5
            assert self.log_diag.ndim == 5
            assert self.off_diag.ndim == 5
        else:
            assert self.mean.ndim == 4
            assert self.log_diag.ndim == 4
            assert self.off_diag.ndim == 4
        
        assert self.mean.shape == self.log_diag.shape
        assert self.mean.device == self.log_diag.device == self.off_diag.device

    def get_num_ch(self):
        return self.log_diag.shape[1]

    def test_consistency(self):
        """
        Test for consistency of the data
        """
        if self.use_3d:
            assert self.log_diag.ndim == 5
            assert self.off_diag.ndim == 5
            num_F = self.off_diag.shape[-4]
        else:
            assert self.log_diag.ndim == 4
            assert self.off_diag.ndim == 4
            num_F = self.off_diag.shape[-3]

        assert self.log_diag.device == self.off_diag.device

        assert self.log_diag.shape[1] > 0

        if self.cross_ch is not None:
            assert self.cross_ch.shape[1] == (self.log_diag.shape[1]**2 - self.log_diag.shape[1]) // 2
            assert self.off_diag.shape[1] == self.log_diag.shape[1] * get_num_off_diag_weights(self.local_connection_dist, self.use_3d)


def get_num_off_diag_weights(local_connection_dist, use_3d=False):
    """Returns the number of off-diagonal entries required for a particular sparsity.

    Args:
        local_connection_dist: Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_3d: Create 3D filters (i.e. 3x3x3) rather than 2D. Defaults to False.

    Returns:
        num_weigts_required (int)
    """
    filter_size = 2 * local_connection_dist + 1
    if use_3d:
        filter_size_dims = filter_size * filter_size * filter_size
    else:
        filter_size_dims = filter_size * filter_size
    filter_size_dims_2 = filter_size_dims // 2
    return filter_size_dims_2


def get_num_cross_channel_weights(num_channels):
    """Returns the number of cross-channel weights required for a particular number of channels. 
    This connects each channel to every other one.

    Args:
        num_channels: The number of channels.

    Returns:
        num_weights_required (int)
    """
    return (num_channels**2 - num_channels) // 2


def convert_log_to_diag_weights(log_diag_weights):
    """Converts the log weight values into the actual positive diagonal values.

    Args:
        log_diag_weights(tensor): [BATCH x 1 x W x H] log of the diagonal terms (mapped through exp).

    Returns:
        diag_weights(tensor): [BATCH x 1 x W x H] actual weights (guaranteed positive)
    """
    diag_values = torch.exp(log_diag_weights)
    return diag_values