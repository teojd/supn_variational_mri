import torch
from torch.distributions import Normal
from dataclasses import dataclass

@dataclass
class Parent:
    def __post_init__(self):
        for (name, field_type) in self.__annotations__.items():
            if not isinstance(self.__dict__[name], field_type):
                current_type = type(self.__dict__[name])
                raise TypeError(f"The field `{name}` was assigned by `{current_type}` instead of `{field_type}`")

@dataclass
class LatentData(Parent):
    mean: torch.Tensor
    log_var: torch.Tensor

    def __post_init__(self):
        if self.mean.shape != self.log_var.shape:
            raise TypeError(f"Mean and variance shapes do not match")
        if self.mean.ndim != 2:
            raise TypeError(f"mean and variance should be 2d torch tensors")

class LogScaleNormal(Normal):
    """
    SUPN distribution class implementing the torch.distributions.Distribution interface.
    """
    
    def __init__(self, latent_data: LatentData):
        self.loc = latent_data.mean
        self.log_var = latent_data.log_var
        self.var = torch.exp(self.log_var)
        self.scale = torch.sqrt(self.var)
        #self.stddev = self.scale
        super().__init__(self.loc, self.scale)

# Register the distribution
torch.distributions.LogScaleNormal = LogScaleNormal



