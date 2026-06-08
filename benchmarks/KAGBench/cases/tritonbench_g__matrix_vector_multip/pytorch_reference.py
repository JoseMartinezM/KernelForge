import torch


def mv(inp: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return torch.mv(inp, vec)
