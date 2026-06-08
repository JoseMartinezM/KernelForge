import torch

from pytorch_reference import _swiglu_fwd as reference_swiglu_fwd


def _candidate_fn(candidate):
    return getattr(candidate, "_swiglu_fwd", candidate)


def _assert_close(actual, expected, *, rtol=1e-5, atol=1e-6):
    assert actual.shape == expected.shape
    assert actual.device == expected.device
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol)


def unit_tests(candidate):
    fn = _candidate_fn(candidate)
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    # Non-power-of-two feature dimension.
    torch.manual_seed(201)
    xy = torch.randn(5, 74, device="cuda", dtype=torch.float32)
    _assert_close(fn(xy), reference_swiglu_fwd(xy))

    # Higher-rank batch shape should be preserved.
    torch.manual_seed(202)
    xy = torch.randn(2, 3, 74, device="cuda", dtype=torch.float32)
    actual = fn(xy)
    expected = reference_swiglu_fwd(xy)
    assert actual.shape == (2, 3, 37)
    _assert_close(actual, expected)

    # Optional out argument should be populated and returned as the output storage.
    torch.manual_seed(203)
    xy = torch.randn(2, 4, 66, device="cuda", dtype=torch.float32)
    out = torch.empty(2, 4, 33, device="cuda", dtype=torch.float32)
    actual = fn(xy, out=out)
    expected = reference_swiglu_fwd(xy)
    assert actual.data_ptr() == out.data_ptr()
    _assert_close(actual, expected)

    # Float16 should match the reference within lower-precision tolerances.
    torch.manual_seed(204)
    xy = torch.randn(7, 130, device="cuda", dtype=torch.float16)
    actual = fn(xy)
    expected = reference_swiglu_fwd(xy)
    _assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    # Non-contiguous input with a strided final dimension should still work.
    torch.manual_seed(205)
    base = torch.randn(3, 148, device="cuda", dtype=torch.float32)
    xy = base[:, ::2]
    assert xy.shape == (3, 74)
    assert xy.stride(-1) != 1
    _assert_close(fn(xy), reference_swiglu_fwd(xy))
