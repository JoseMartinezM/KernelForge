from typing import Optional

import torch


def logsumexp_fwd(
    x: torch.Tensor,
    scale: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """PyTorch reference for logsumexp over the last dimension."""
    values = x.to(torch.float32)
    if scale is not None:
        values = values * scale

    result = torch.logsumexp(values, dim=-1)
    if dtype is not None and dtype != torch.float and dtype != torch.float32:
        result = result.to(dtype=dtype)
    return result
