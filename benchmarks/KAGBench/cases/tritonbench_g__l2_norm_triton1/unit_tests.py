import torch

from pytorch_reference import _l2_norm_fwd as reference_l2_norm_fwd


def _assert_matches_reference(candidate_fn, x, eps=1e-6, rtol=1e-5, atol=1e-6):
    actual = candidate_fn(x, eps=eps)
    expected = reference_l2_norm_fwd(x, eps=eps)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def unit_tests(candidate):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these tests")

    candidate_fn = candidate._l2_norm_fwd

    # Odd feature size with a two-dimensional float32 tensor.
    torch.manual_seed(301)
    x = torch.randn(5, 7, device="cuda", dtype=torch.float32)
    _assert_matches_reference(candidate_fn, x)

    # All-zero rows should stay zero because eps prevents division by zero.
    x = torch.zeros(3, 9, device="cuda", dtype=torch.float32)
    _assert_matches_reference(candidate_fn, x)

    # Explicit negative and mixed-sign values with a non-default epsilon.
    x = torch.tensor(
        [[-3.0, -4.0, 0.0], [1.5, -2.5, 3.5]],
        device="cuda",
        dtype=torch.float32,
    )
    _assert_matches_reference(candidate_fn, x, eps=1e-4)

    # One-dimensional input is treated as a single row and reshaped back.
    torch.manual_seed(302)
    x = torch.randn(11, device="cuda", dtype=torch.float32)
    _assert_matches_reference(candidate_fn, x)

    # Higher-rank shape variant: normalize over the last dimension only.
    torch.manual_seed(303)
    x = torch.randn(2, 3, 5, device="cuda", dtype=torch.float32)
    _assert_matches_reference(candidate_fn, x)

    # Non-contiguous logical input with stride in the last dimension not equal to 1.
    torch.manual_seed(304)
    base = torch.randn(7, 4, device="cuda", dtype=torch.float32)
    x = base.t()
    assert not x.is_contiguous()
    _assert_matches_reference(candidate_fn, x)

    # Float16 input, if supported by the candidate/runtime, should return float16 values.
    torch.manual_seed(305)
    x = torch.randn(4, 13, device="cuda", dtype=torch.float16)
    _assert_matches_reference(candidate_fn, x, rtol=2e-3, atol=2e-3)

    # Float16 zeros cover the small-norm path in reduced precision.
    x = torch.zeros(2, 5, device="cuda", dtype=torch.float16)
    _assert_matches_reference(candidate_fn, x, rtol=2e-3, atol=2e-3)
