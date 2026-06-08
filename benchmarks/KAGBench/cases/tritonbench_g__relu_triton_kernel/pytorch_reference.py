import torch


def relu(x: torch.Tensor) -> torch.Tensor:
    """Reference implementation for the benchmark public API."""
    return torch.relu(x).to(dtype=torch.float32)
