import torch


def triton_mul2(x: torch.Tensor, BLOCK_SIZE: int = 16) -> torch.Tensor:
    """Reference implementation for returning a doubled copy of x."""
    return 2 * x


def triton_mul2_inplace(x: torch.Tensor, BLOCK_SIZE: int = 16) -> torch.Tensor:
    """Reference implementation for doubling x in place."""
    x.mul_(2)
    return x
