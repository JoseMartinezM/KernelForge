import torch

import pytorch_reference as reference


def _check_add(
    candidate,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    seed: int,
    *,
    rtol: float,
    atol: float,
) -> None:
    torch.manual_seed(seed)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = torch.randn(shape, device="cuda", dtype=dtype)
    x_before = x.clone()
    y_before = y.clone()

    expected = reference.add_wrapper(x, y)
    actual = candidate.add_wrapper(x, y)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device.type == "cuda"
    if actual.numel() > 0:
        assert actual.data_ptr() != x.data_ptr()
        assert actual.data_ptr() != y.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(x, x_before, rtol=0, atol=0)
    torch.testing.assert_close(y, y_before, rtol=0, atol=0)


def unit_tests(candidate) -> None:
    _check_add(candidate, (17,), torch.float32, 11, rtol=1e-5, atol=1e-6)
    _check_add(candidate, (0,), torch.float32, 12, rtol=1e-5, atol=1e-6)
    _check_add(candidate, (2, 3, 5), torch.float32, 13, rtol=1e-5, atol=1e-6)
    _check_add(candidate, (33,), torch.float16, 14, rtol=1e-3, atol=1e-3)
