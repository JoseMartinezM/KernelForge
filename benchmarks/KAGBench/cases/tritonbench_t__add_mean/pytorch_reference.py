import torch


def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    result = input + alpha * other
    return result.mean(dim=dim, keepdim=keepdim, dtype=dtype)
