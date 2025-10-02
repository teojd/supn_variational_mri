import torch
from torch import nn

from supn_base.supn_distribution import SUPN

def log_likelihood_loss(x: torch.tensor,
                        model: nn.Module,
                        supn_dist: SUPN = None) -> torch.tensor:
    """
    Compute the log likelihood loss.

    Args:
        supn_dist (SUPN): SUPN torch.distribution object.
        x (torch.Tensor): Data tensor.

    Returns:
        torch.Tensor: Log likelihood of x under the model.
    """
    if supn_dist is None:
        supn_dist, _ = model.forward(x)

    LogProb = supn_dist.log_prob(x).sum()
    return -LogProb, -LogProb

def VAE_loss(x: torch.tensor,
             model: nn.Module) -> torch.tensor:
    """
    Compute the VAE loss function.
    
    Args:
        x (torch.Tensor): Data tensor.
        model (nn.Module): SUPN model class.
    
    Returns:
        torch.Tensor: Total loss (Log likelihood loss + KL divergence).
    """
    supn_dist, latent_dist = model.forward(x)

    NLL, _ = log_likelihood_loss(x = x, 
                              model = None, 
                              supn_dist = supn_dist)
    KLDiv = 0.5 * (latent_dist.var + latent_dist.mean**2 - latent_dist.log_var).sum()
    return NLL + KLDiv, NLL