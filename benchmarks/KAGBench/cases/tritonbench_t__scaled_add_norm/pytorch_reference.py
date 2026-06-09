import torch


def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    y += alpha * x
    return torch.norm(y)
