import torch

from pytorch_reference import rms_norm as reference_rms_norm


def _assert_matches_reference(candidate, x: torch.Tensor, weight: torch.Tensor, *, eps=1e-5, rtol=1e-5, atol=1e-6) -> None:
    normalized_shape = (x.shape[-1],)
    actual = candidate.rms_norm(x, normalized_shape, weight, eps=eps)
    expected = reference_rms_norm(x, normalized_shape, weight, eps=eps)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def public_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"

    torch.manual_seed(101)
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    weight = torch.randn(8, device="cuda", dtype=torch.float32)
    _assert_matches_reference(candidate, x, weight)

    torch.manual_seed(102)
    x = torch.randn(2, 3, 7, device="cuda", dtype=torch.float32)
    weight = torch.randn(7, device="cuda", dtype=torch.float32)
    _assert_matches_reference(candidate, x, weight, eps=1e-4)
