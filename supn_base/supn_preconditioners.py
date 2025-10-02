import torch
from supn_data import SUPNData

class SUPNPreconditionerBase(torch.nn.Module):
    # This base class is used to define the interface for the SUPN preconditioner classes
    def __init__(self):
        super().__init__()


    def forward(self, supn_data: SUPNData):
        raise NotImplementedError("Forward not implemented for class SUPNPreconditionerBase(torch.nn.Module")
    

class SUPNScalarPreconditioner(SUPNPreconditionerBase):
    # This class scales the individual parameters of the SUPN data to provide values in an initially reasonable range and prevent excessive values
    def __init__(self, init_std_dev: float, init_off_diag_scal: float, num_local_connections: int, use_3d: bool):
        from numpy import log
        super().__init__()

        self._initial_log_diag_offset = -log(init_std_dev)
        self._off_diag_scale = init_off_diag_scal

    def forward(self, supn_data: SUPNData):
        supn_data.log_diag = supn_data.log_diag + self._initial_log_diag_offset
        supn_data.cross_ch = supn_data.cross_ch * self._off_diag_scale
        supn_data.off_diag = supn_data.off_diag * self._off_diag_scale
        return supn_data
