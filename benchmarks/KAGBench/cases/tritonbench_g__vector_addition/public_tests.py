import torch

from pytorch_reference import add as reference_add


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4), (actual, expected)


def public_tests(candidate):
    torch.manual_seed(0)

    x = torch.rand(1024, device="cuda", dtype=torch.float32)
    y = torch.rand(1024, device="cuda", dtype=torch.float32)
    _assert_close(candidate.add(x, y), reference_add(x, y))

    torch.manual_seed(1)
    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    y = torch.randn(4096, device="cuda", dtype=torch.float32)
    _assert_close(candidate.add(x, y), reference_add(x, y))
