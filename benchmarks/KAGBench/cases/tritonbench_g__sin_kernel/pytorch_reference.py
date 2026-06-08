import torch


def call_kernel(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation for the public call_kernel(x) API."""
    return torch.sin(x)
