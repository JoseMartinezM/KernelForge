import torch

from pytorch_reference import kldivergence as reference_kldivergence


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.is_cuda
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def public_tests(candidate) -> None:
    torch.manual_seed(1234)
    x = torch.rand(1024, device="cuda", dtype=torch.float32) + 0.1
    y = torch.rand(1024, device="cuda", dtype=torch.float32) + 0.1
    _assert_close(candidate.kldivergence(x, y), reference_kldivergence(x, y))

    torch.manual_seed(5678)
    x = torch.rand((64, 32), device="cuda", dtype=torch.float32) + 0.05
    y = torch.rand((64, 32), device="cuda", dtype=torch.float32) + 0.05
    _assert_close(candidate.kldivergence(x, y), reference_kldivergence(x, y))
