import torch


def dropout(x: torch.Tensor, x_keep: torch.Tensor, p: float) -> torch.Tensor:
    """PyTorch reference for dropout with a caller-provided keep mask."""
    return torch.where(x_keep.bool(), x / (1.0 - p), torch.zeros_like(x))
