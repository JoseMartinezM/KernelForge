import torch

from pytorch_reference import rms_norm as reference_rms_norm


def _assert_matches_reference(candidate, x: torch.Tensor, weight: torch.Tensor, *, eps=1e-5, rtol=1e-5, atol=1e-6) -> None:
    normalized_shape = (x.shape[-1],)
    actual = candidate.rms_norm(x, normalized_shape, weight, eps=eps)
    expected = reference_rms_norm(x, normalized_shape, weight, eps=eps)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def unit_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"

    # Two-dimensional float32 input with an odd feature size.
    torch.manual_seed(201)
    x = torch.randn(5, 13, device="cuda", dtype=torch.float32)
    weight = torch.randn(13, device="cuda", dtype=torch.float32)
    _assert_matches_reference(candidate, x, weight)

    # Explicit negative and mixed-sign values with a non-default epsilon.
    x = torch.tensor(
        [[-3.0, -1.0, 0.5, 2.0, -4.5], [1.25, -2.5, 3.75, -4.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )
    weight = torch.tensor([1.0, -0.5, 2.0, 0.25, -1.5], device="cuda", dtype=torch.float32)
    _assert_matches_reference(candidate, x, weight, eps=1e-3)

    # Three-dimensional float32 input normalizes over the last dimension only.
    torch.manual_seed(202)
    x = torch.randn(2, 4, 11, device="cuda", dtype=torch.float32)
    weight = torch.randn(11, device="cuda", dtype=torch.float32)
    _assert_matches_reference(candidate, x, weight, eps=1e-6)

    # Float16 two-dimensional input uses looser tolerances for reduced precision.
    torch.manual_seed(203)
    x = torch.randn(4, 9, device="cuda", dtype=torch.float16)
    weight = torch.randn(9, device="cuda", dtype=torch.float16)
    _assert_matches_reference(candidate, x, weight, rtol=2e-3, atol=2e-3)

    # Float16 three-dimensional input with an odd feature size and larger epsilon.
    torch.manual_seed(204)
    x = torch.randn(2, 3, 7, device="cuda", dtype=torch.float16)
    weight = torch.randn(7, device="cuda", dtype=torch.float16)
    _assert_matches_reference(candidate, x, weight, eps=1e-3, rtol=2e-3, atol=2e-3)
