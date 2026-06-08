import torch

from pytorch_reference import _swiglu_fwd as reference_swiglu_fwd


def _candidate_fn(candidate):
    return getattr(candidate, "_swiglu_fwd", candidate)


def public_tests(candidate):
    fn = _candidate_fn(candidate)
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    torch.manual_seed(101)
    xy = torch.randn(4, 256, device="cuda", dtype=torch.float32)
    actual = fn(xy)
    expected = reference_swiglu_fwd(xy)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)

    torch.manual_seed(102)
    xy = torch.randn(2, 64, device="cuda", dtype=torch.float32)
    out = torch.empty(2, 32, device="cuda", dtype=torch.float32)
    actual = fn(xy, out=out)
    expected = reference_swiglu_fwd(xy)
    assert actual.data_ptr() == out.data_ptr()
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
