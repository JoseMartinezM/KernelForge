import torch


def _swiglu_fwd(xy, out=None):
    """PyTorch reference for SwiGLU forward on the last dimension."""
    x, y = xy.chunk(2, dim=-1)
    result = x * torch.sigmoid(x) * y
    if out is not None:
        out.copy_(result)
        return out
    return result
