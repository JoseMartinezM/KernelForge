import torch

from pytorch_reference import custom_add as reference_custom_add


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float16:
        return 1e-2, 1e-2
    return 1e-5, 1e-5


def _check(candidate, a: torch.Tensor, b: torch.Tensor) -> None:
    expected = reference_custom_add(a, b)
    actual = candidate.custom_add(a, b)
    rtol, atol = _tolerances(a.dtype)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (actual, expected)
    if actual.numel() > 0:
        assert actual.data_ptr() != a.data_ptr()
        assert actual.data_ptr() != b.data_ptr()


def public_tests(candidate):
    torch.manual_seed(101)
    a = torch.randn(17, device="cuda", dtype=torch.float32) - 3.0
    b = torch.randn(17, device="cuda", dtype=torch.float32) * 2.0
    _check(candidate, a, b)

    torch.manual_seed(102)
    a = torch.randn((3, 5), device="cuda", dtype=torch.float16)
    b = torch.randn((3, 5), device="cuda", dtype=torch.float16)
    _check(candidate, a, b)
