import torch

import pytorch_reference as reference


def _assert_add_matches(candidate, x: torch.Tensor, y: torch.Tensor, *, rtol: float, atol: float) -> None:
    expected = reference.add_wrapper(x, y)
    actual = candidate.add_wrapper(x, y)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device.type == "cuda"
    if actual.numel() > 0:
        assert actual.data_ptr() != x.data_ptr()
        assert actual.data_ptr() != y.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def public_tests(candidate) -> None:
    torch.manual_seed(0)
    x = torch.randn(16, device="cuda", dtype=torch.float32)
    y = torch.randn(16, device="cuda", dtype=torch.float32)
    _assert_add_matches(candidate, x, y, rtol=1e-5, atol=1e-6)

    torch.manual_seed(1)
    x = torch.randn((4, 8), device="cuda", dtype=torch.float32)
    y = torch.randn((4, 8), device="cuda", dtype=torch.float32)
    _assert_add_matches(candidate, x, y, rtol=1e-5, atol=1e-6)
