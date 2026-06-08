import torch


def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    """PyTorch reference for last-dimension softmax with an optional additive mask."""
    return torch.softmax(input if mask is None else input + mask, dim=-1)
