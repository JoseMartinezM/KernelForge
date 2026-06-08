import torch


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Reference row-wise softmax over the second dimension."""
    return torch.softmax(x, dim=1)
