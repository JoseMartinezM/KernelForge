import torch

from pytorch_reference import square as reference_square


def _assert_matches_reference(
    candidate,
    x: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    original = x.clone()
    expected = reference_square(x)
    actual = candidate.square(x)

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.device == x.device
    assert actual.data_ptr() != x.data_ptr()
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def unit_tests(candidate) -> None:
    torch.manual_seed(2001)
    float32_values = torch.randn(5, 17, device="cuda", dtype=torch.float32) * 4.0 - 2.0
    _assert_matches_reference(candidate, float32_values.contiguous(), rtol=1e-6, atol=1e-6)

    torch.manual_seed(2002)
    odd_columns = torch.randn(3, 129, device="cuda", dtype=torch.float32) - 0.5
    _assert_matches_reference(candidate, odd_columns.contiguous(), rtol=1e-6, atol=1e-6)

    torch.manual_seed(2003)
    float16_values = (torch.randn(7, 37, device="cuda", dtype=torch.float16) * 3.0) - 1.5
    _assert_matches_reference(candidate, float16_values.contiguous(), rtol=1e-3, atol=1e-3)
