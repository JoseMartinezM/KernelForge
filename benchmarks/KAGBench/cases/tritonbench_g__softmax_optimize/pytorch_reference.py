import torch


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Reference row-wise softmax over dim=1."""
    return torch.softmax(x, dim=1)
