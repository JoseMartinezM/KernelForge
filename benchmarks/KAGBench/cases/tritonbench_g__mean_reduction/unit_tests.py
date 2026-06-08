import torch

from pytorch_reference import mean_dim as reference_mean_dim


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> None:
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"dtype mismatch: {actual.dtype} != {expected.dtype}"
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _run_case(candidate, x: torch.Tensor, dim, keepdim: bool = False, *, dtype=None, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    expected = reference_mean_dim(x, dim, keepdim=keepdim, dtype=dtype)
    actual = candidate.mean_dim(x, dim, keepdim=keepdim, dtype=dtype)
    _assert_close(actual, expected, atol=atol, rtol=rtol)


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    # Rectangular shape, reduce one middle axis.
    torch.manual_seed(201)
    x = torch.randn(5, 7, 3, device="cuda", dtype=torch.float32)
    _run_case(candidate, x, 1)

    # Odd dimensions and two-axis reduction with dimensions kept.
    torch.manual_seed(202)
    x = torch.randn(3, 5, 7, 9, device="cuda", dtype=torch.float32)
    _run_case(candidate, x, [1, 3], keepdim=True)

    # Negative values and negative axis.
    torch.manual_seed(203)
    x = torch.randn(4, 6, 5, device="cuda", dtype=torch.float32) * 3.0 - 2.0
    _run_case(candidate, x, -1)

    # Float16 input reduced over a nontrivial odd axis.
    torch.manual_seed(204)
    x = (torch.randn(2, 9, 5, device="cuda", dtype=torch.float16) - 0.5) * 2.0
    _run_case(candidate, x, 1, atol=2e-2, rtol=2e-2)

    # Float16 input with float32 output requested.
    torch.manual_seed(205)
    x = torch.randn(3, 5, 7, device="cuda", dtype=torch.float16)
    _run_case(candidate, x, [1, 2], dtype=torch.float32, atol=2e-3, rtol=2e-3)

    # Float32 reduction over multiple rectangular axes without keepdim.
    torch.manual_seed(206)
    x = torch.randn(2, 3, 8, 5, device="cuda", dtype=torch.float32)
    _run_case(candidate, x, [0, 2])
