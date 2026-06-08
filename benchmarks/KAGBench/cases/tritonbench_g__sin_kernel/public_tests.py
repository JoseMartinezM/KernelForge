import torch

from pytorch_reference import call_kernel as reference_call_kernel


def _assert_matches_reference(candidate, x: torch.Tensor, *, rtol: float, atol: float) -> None:
    actual = candidate.call_kernel(x)
    expected = reference_call_kernel(x)

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def public_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    values = torch.tensor([0.0, 0.5, 1.0, 2.0], device="cuda", dtype=torch.float32)
    _assert_matches_reference(candidate, values, rtol=1e-5, atol=1e-6)

    torch.manual_seed(101)
    matrix = (torch.rand((16, 32), device="cuda", dtype=torch.float32) * 6.0 - 3.0).contiguous()
    _assert_matches_reference(candidate, matrix, rtol=1e-4, atol=1e-4)
