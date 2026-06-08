import torch


def log_softmax(x: torch.Tensor, dim=-1, dtype=None) -> torch.Tensor:
    """Reference implementation using PyTorch log_softmax."""
    return torch.nn.functional.log_softmax(x, dim=dim, dtype=dtype)
