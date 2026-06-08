import torch


def masked_add(
    grad: torch.Tensor,
    p_data: torch.Tensor,
    p_mask: torch.Tensor,
    alpha: float = 0,
):
    """Reference implementation for masked in-place addition."""
    update_mask = ~p_mask.to(torch.bool)
    grad.add_(p_data * update_mask.to(dtype=p_data.dtype), alpha=alpha)
    return grad
