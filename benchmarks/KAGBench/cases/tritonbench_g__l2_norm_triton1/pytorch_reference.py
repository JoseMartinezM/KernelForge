import torch


def _l2_norm_fwd(x, eps=1e-6):
    """PyTorch reference for row-wise L2 normalization over the last dimension."""
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    x_float = x_2d.to(torch.float32)
    denom = torch.sqrt(torch.sum(x_float * x_float, dim=-1, keepdim=True) + eps)
    y = x_float / denom
    return y.to(dtype=x.dtype).reshape(original_shape)
