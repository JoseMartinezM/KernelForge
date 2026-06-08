import torch

from pytorch_reference import custom_add as reference_custom_add


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float16:
        return 1e-2, 1e-2
    return 1e-5, 1e-5


def _check(candidate, a: torch.Tensor, b: torch.Tensor) -> None:
    expected = reference_custom_add(a, b)
    a_before = a.clone()
    b_before = b.clone()
    actual = candidate.custom_add(a, b)
    rtol, atol = _tolerances(a.dtype)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (actual, expected)
    assert torch.allclose(a, a_before, rtol=rtol, atol=atol)
    assert torch.allclose(b, b_before, rtol=rtol, atol=atol)
    if actual.numel() > 0:
        assert actual.data_ptr() != a.data_ptr()
        assert actual.data_ptr() != b.data_ptr()


def _check_random(candidate, shape, dtype: torch.dtype, seed: int, scale: float = 1.0) -> None:
    torch.manual_seed(seed)
    a = torch.randn(shape, device="cuda", dtype=dtype) * scale
    b = torch.randn(shape, device="cuda", dtype=dtype) * -scale
    _check(candidate, a.contiguous(), b.contiguous())


def unit_tests(candidate):
    _check(candidate, torch.empty(0, device="cuda", dtype=torch.float32), torch.empty(0, device="cuda", dtype=torch.float32))

    _check_random(candidate, shape=(1,), dtype=torch.float32, seed=201)
    _check_random(candidate, shape=(31,), dtype=torch.float32, seed=202, scale=5.0)
    _check_random(candidate, shape=(513,), dtype=torch.float32, seed=203)
    _check_random(candidate, shape=(4099,), dtype=torch.float32, seed=204, scale=0.25)

    torch.manual_seed(205)
    a = torch.linspace(-8.0, 8.0, 67, device="cuda", dtype=torch.float32)
    b = torch.linspace(3.5, -3.5, 67, device="cuda", dtype=torch.float32)
    _check(candidate, a, b)

    _check_random(candidate, shape=(129,), dtype=torch.float16, seed=206, scale=2.0)
    _check_random(candidate, shape=(1025,), dtype=torch.float16, seed=207)
