import torch


def sin_triton(x: torch.Tensor, out: torch.Tensor):
    """Reference implementation for sin_triton(x, out)."""
    return torch.sin(x, out=out)
