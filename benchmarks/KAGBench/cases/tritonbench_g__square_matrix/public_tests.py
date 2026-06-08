import torch

from pytorch_reference import square as reference_square


def _assert_matches_reference(candidate, x: torch.Tensor) -> None:
    original = x.clone()
    expected = reference_square(x)
    actual = candidate.square(x)

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.device == x.device
    assert actual.data_ptr() != x.data_ptr()
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def public_tests(candidate) -> None:
    torch.manual_seed(1001)
    _assert_matches_reference(
        candidate,
        torch.randn((16, 16), device="cuda", dtype=torch.float32),
    )

    torch.manual_seed(1002)
    _assert_matches_reference(
        candidate,
        torch.randn((8, 32), device="cuda", dtype=torch.float32),
    )
