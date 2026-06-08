import torch

from pytorch_reference import logsumexp_fwd as reference_logsumexp_fwd


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float = 1e-4, atol: float = 1e-4) -> None:
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"dtype mismatch: {actual.dtype} != {expected.dtype}"
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _check(candidate, x: torch.Tensor, **kwargs) -> None:
    assert x.is_contiguous()
    actual = candidate.logsumexp_fwd(x, **kwargs)
    expected = reference_logsumexp_fwd(x, **kwargs)
    _assert_close(actual, expected)


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    torch.manual_seed(101)
    x = torch.randn(4, 64, device="cuda", dtype=torch.float32)
    _check(candidate, x)

    torch.manual_seed(102)
    y = torch.randn(2, 3, 32, device="cuda", dtype=torch.float32)
    _check(candidate, y, scale=0.5)
