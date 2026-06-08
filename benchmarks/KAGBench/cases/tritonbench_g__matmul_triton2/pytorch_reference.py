import torch


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)
