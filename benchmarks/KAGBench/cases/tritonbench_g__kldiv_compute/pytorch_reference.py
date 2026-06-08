import torch


def kldivergence(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference elementwise KL divergence term."""
    return x * torch.log(x / y)
