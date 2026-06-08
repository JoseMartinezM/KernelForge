import torch

from pytorch_reference import mean_dim as reference_mean_dim


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"dtype mismatch: {actual.dtype} != {expected.dtype}"
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    torch.manual_seed(101)
    x = torch.randn(3, 4, 5, device="cuda", dtype=torch.float32)
    expected = reference_mean_dim(x, 1)
    actual = candidate.mean_dim(x, 1)
    _assert_close(actual, expected)

    torch.manual_seed(102)
    y = torch.randn(2, 3, 4, 5, device="cuda", dtype=torch.float32)
    expected = reference_mean_dim(y, [1, 2], keepdim=True)
    actual = candidate.mean_dim(y, [1, 2], keepdim=True)
    _assert_close(actual, expected)
