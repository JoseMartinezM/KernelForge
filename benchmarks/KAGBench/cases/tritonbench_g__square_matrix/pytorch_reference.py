import torch


def square(x: torch.Tensor) -> torch.Tensor:
    """Return the elementwise square of a two-dimensional tensor."""
    return torch.mul(x, x)
