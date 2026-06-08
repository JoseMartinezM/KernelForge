import torch

from pytorch_reference import _l2_norm_fwd as reference_l2_norm_fwd


def _assert_matches_reference(candidate_fn, x, eps=1e-6, rtol=1e-5, atol=1e-6):
    actual = candidate_fn(x, eps=eps)
    expected = reference_l2_norm_fwd(x, eps=eps)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def public_tests(candidate):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these tests")

    candidate_fn = candidate._l2_norm_fwd

    torch.manual_seed(123)
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    _assert_matches_reference(candidate_fn, x)

    torch.manual_seed(124)
    x = torch.randn(2, 3, 5, device="cuda", dtype=torch.float32)
    _assert_matches_reference(candidate_fn, x)
