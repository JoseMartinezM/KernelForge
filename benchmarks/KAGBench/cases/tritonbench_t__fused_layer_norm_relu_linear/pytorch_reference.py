import torch
import torch.nn.functional as F


def fused_layer_norm_relu_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    normalized_shape: torch.Size | int | tuple[int, ...] | None = None,
    eps: float = 1e-05,
    elementwise_affine: bool = True,
) -> torch.Tensor:
    linear_output = F.linear(input, weight, bias)
    relu_output = F.relu(linear_output)
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    return F.layer_norm(relu_output, normalized_shape, eps=eps)
