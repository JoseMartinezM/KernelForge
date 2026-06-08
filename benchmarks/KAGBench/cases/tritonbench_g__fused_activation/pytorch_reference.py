import torch


def fused_add_mul_activation_torch(
    in_out_tensor: torch.Tensor,
    bias: torch.Tensor,
    in_tensor: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for the in-place fused add, multiply, and sigmoid activation."""
    bias_indices = torch.arange(in_out_tensor.numel(), device=in_out_tensor.device) % bias.numel()
    result = torch.sigmoid(in_out_tensor + bias[bias_indices] + 0.5 * in_tensor)
    in_out_tensor.copy_(result)
    return in_out_tensor
