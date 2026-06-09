import torch
from torch import Tensor


def normalized_cosine_similarity(
    x1: Tensor,
    x2: Tensor,
    dim: int = 1,
    eps_similarity: float = 1e-08,
    p_norm: float = 2,
    eps_norm: float = 1e-12,
) -> Tensor:
    x1_normalized = torch.nn.functional.normalize(x1, p=p_norm, dim=dim, eps=eps_norm)
    x2_normalized = torch.nn.functional.normalize(x2, p=p_norm, dim=dim, eps=eps_norm)
    return torch.nn.functional.cosine_similarity(
        x1_normalized, x2_normalized, dim=dim, eps=eps_similarity
    )
