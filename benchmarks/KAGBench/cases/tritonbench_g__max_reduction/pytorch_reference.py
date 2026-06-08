from collections import namedtuple

import torch


MaxResult = namedtuple("max", ["values", "indices"])


def max(inp: torch.Tensor) -> torch.Tensor:
    """Reference implementation for overall max reduction using PyTorch."""
    return torch.max(inp)


def max_dim(inp: torch.Tensor, dim=None, keepdim: bool = False):
    """Reference implementation for dimension-wise max reduction using PyTorch."""
    if dim is None:
        result = torch.max(inp.reshape(-1), dim=0, keepdim=keepdim)
        values = result.values
        indices = result.indices
        if keepdim:
            keep_shape = (1,) * inp.ndim
            values = values.reshape(keep_shape)
            indices = indices.reshape(keep_shape)
        return MaxResult(values=values, indices=indices)

    result = torch.max(inp, dim=dim, keepdim=keepdim)
    return MaxResult(values=result.values, indices=result.indices)
