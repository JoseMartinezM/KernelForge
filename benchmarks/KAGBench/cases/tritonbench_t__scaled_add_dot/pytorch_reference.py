import torch


def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    y += alpha * x
    return torch.dot(y, y)
