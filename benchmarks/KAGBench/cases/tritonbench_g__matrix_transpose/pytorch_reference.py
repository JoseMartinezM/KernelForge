import torch


def wrapper(size_m: int, d_head: int) -> torch.Tensor:
    matrix = torch.randn((size_m, d_head), dtype=torch.float16, device="cuda")
    return matrix.transpose(0, 1).contiguous()
