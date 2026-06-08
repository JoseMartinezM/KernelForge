import torch

from pytorch_reference import geglu_backward as reference_geglu_backward
from pytorch_reference import geglu_forward as reference_geglu_forward


def _assert_same_metadata(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float) -> None:
    _assert_same_metadata(actual, expected)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _check_case(candidate, a: torch.Tensor, b: torch.Tensor, dc: torch.Tensor) -> None:
    assert a.is_cuda and b.is_cuda and dc.is_cuda
    assert a.shape == b.shape == dc.shape
    assert a.ndim == 2 and a.shape[-1] == 128
    assert a.is_contiguous() and b.is_contiguous() and dc.is_contiguous()

    a_before = a.clone()
    b_before = b.clone()

    expected_a, expected_b, expected_c = reference_geglu_forward(a, b)
    actual_a, actual_b, actual_c = candidate.geglu_forward(a, b)

    _assert_close(actual_a, expected_a, rtol=0.0, atol=0.0)
    _assert_close(actual_b, expected_b, rtol=0.0, atol=0.0)
    _assert_close(a, a_before, rtol=0.0, atol=0.0)
    _assert_close(b, b_before, rtol=0.0, atol=0.0)
    _assert_close(actual_c, expected_c, rtol=1e-5, atol=1e-5)

    expected_da, expected_db = reference_geglu_backward(a_before, b_before, dc)
    actual_da, actual_db = candidate.geglu_backward(a_before.clone(), b_before.clone(), dc.clone())

    _assert_close(actual_da, expected_da, rtol=1e-5, atol=2e-5)
    _assert_close(actual_db, expected_db, rtol=1e-5, atol=1e-5)


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    a = torch.linspace(-3.0, 2.5, steps=2 * 128, device="cuda", dtype=torch.float32).reshape(2, 128).contiguous()
    b = torch.linspace(1.5, -2.0, steps=2 * 128, device="cuda", dtype=torch.float32).reshape(2, 128).contiguous()
    dc = torch.linspace(-1.25, 1.25, steps=2 * 128, device="cuda", dtype=torch.float32).reshape(2, 128).contiguous()
    _check_case(candidate, a, b, dc)

    torch.manual_seed(101)
    a = (torch.randn(5, 128, device="cuda", dtype=torch.float32) * 1.75 - 0.5).contiguous()
    b = (torch.randn(5, 128, device="cuda", dtype=torch.float32) * 1.25).contiguous()
    dc = (torch.randn(5, 128, device="cuda", dtype=torch.float32) * 0.75 - 0.25).contiguous()
    _check_case(candidate, a, b, dc)
