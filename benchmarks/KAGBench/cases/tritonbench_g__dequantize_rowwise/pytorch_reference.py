import torch


def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor) -> torch.Tensor:
    """Reference row-wise dequantization using PyTorch operations."""
    scaled = state_x.to(torch.float32).unsqueeze(1) * x.to(torch.float32)
    return (scaled / 127.0).to(torch.float16)
