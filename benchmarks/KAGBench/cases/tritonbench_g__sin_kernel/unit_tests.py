import torch

from pytorch_reference import call_kernel as reference_call_kernel


def _assert_matches_reference(
    candidate,
    x: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    original = x.clone()
    actual = candidate.call_kernel(x)
    expected = reference_call_kernel(x)

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"dtype mismatch: {actual.dtype} != {expected.dtype}"
    assert actual.device == expected.device, f"device mismatch: {actual.device} != {expected.device}"
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


def unit_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    # Negative, zero, and positive values in a non-power-of-two length.
    values = torch.tensor(
        [-6.0, -3.1415927, -1.0, -0.0, 0.0, 0.25, 1.0, 3.1415927, 6.0],
        device="cuda",
        dtype=torch.float32,
    )
    _assert_matches_reference(candidate, values, rtol=1e-4, atol=1e-4)

    # Zero-length contiguous CUDA tensor should return an empty tensor of the same dtype.
    empty = torch.empty((0,), device="cuda", dtype=torch.float32)
    _assert_matches_reference(candidate, empty, rtol=0.0, atol=0.0)

    # Odd one-dimensional length, larger than common block sizes.
    torch.manual_seed(201)
    odd_length = (torch.randn(1031, device="cuda", dtype=torch.float32) * 8.0 - 4.0).contiguous()
    _assert_matches_reference(candidate, odd_length, rtol=1e-4, atol=1e-4)

    # Higher-rank contiguous float32 input.
    torch.manual_seed(202)
    volume = (torch.randn((3, 5, 7), device="cuda", dtype=torch.float32) * 5.0).contiguous()
    _assert_matches_reference(candidate, volume, rtol=1e-4, atol=1e-4)

    # Float16 input with lower-precision tolerance.
    torch.manual_seed(203)
    half_values = (torch.randn((11, 17), device="cuda", dtype=torch.float16) * 4.0).contiguous()
    _assert_matches_reference(candidate, half_values, rtol=2e-3, atol=2e-3)
