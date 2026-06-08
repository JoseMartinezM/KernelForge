import torch


def cos(A):
    """Reference implementation for the public cos(A) API."""
    return torch.cos(A.to(torch.float32)).to(dtype=A.dtype)
