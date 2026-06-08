import torch


def mean_dim(x: torch.Tensor, dim, keepdim: bool = False, *, dtype=None) -> torch.Tensor:
    """Reference implementation for mean_dim using PyTorch."""
    return torch.mean(x, dim=dim, keepdim=keepdim, dtype=dtype)
