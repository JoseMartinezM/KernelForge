import torch

from pytorch_reference import puzzle1 as reference_puzzle1


def _check(candidate, x: torch.Tensor) -> None:
    actual = candidate.puzzle1(x)
    expected = reference_puzzle1(x)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.allclose(actual, expected)


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    torch.manual_seed(101)
    _check(candidate, torch.randn(16, device="cuda", dtype=torch.float32))

    torch.manual_seed(102)
    _check(candidate, torch.randn(4, 8, device="cuda", dtype=torch.float32))
