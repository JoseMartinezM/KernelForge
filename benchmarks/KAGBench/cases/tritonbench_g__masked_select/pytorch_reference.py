import torch


def masked_select(inp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reference implementation for the public masked_select API."""
    return torch.masked_select(inp, mask)
