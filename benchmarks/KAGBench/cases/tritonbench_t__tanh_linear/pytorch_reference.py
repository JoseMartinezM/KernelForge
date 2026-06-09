import torch


def tanh_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    output = input @ weight.t()
    if bias is not None:
        output = output + bias
    return torch.tanh(output)
