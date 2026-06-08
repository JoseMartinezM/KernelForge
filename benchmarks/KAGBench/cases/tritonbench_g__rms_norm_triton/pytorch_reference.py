import torch


def rms_norm(x: torch.Tensor, normalized_shape, weight: torch.Tensor, eps=1e-5) -> torch.Tensor:
    """PyTorch reference for RMS normalization over the last dimension."""
    x_float = x.to(torch.float32)
    mean_square = torch.mean(x_float * x_float, dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + eps)
    normalized = (x_float * inv_rms).to(dtype=x.dtype)
    return (normalized * weight).to(dtype=x.dtype)
