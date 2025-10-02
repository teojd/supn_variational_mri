import torch
import torch.nn.functional as F

def softclip(tensor, min = -5, max = 5):
    """
    Smoothly clips the input tensor to a minimum value.

    Args:
        tensor (torch.Tensor): Input tensor.
        min (float, optional): Minimum value to clip to. 

    Returns:
        torch.Tensor: Clipped tensor.
    """
    result_tensor = min + F.softplus(tensor - min) - F.softplus(tensor - max)

    return result_tensor
