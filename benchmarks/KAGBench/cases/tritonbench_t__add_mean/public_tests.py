import torch
from pytorch_reference import add_mean as reference_add_mean


def _assert_close(actual, expected, *, rtol=1e-5, atol=1e-6):
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (actual, expected)


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
    other = torch.full_like(x, 0.5)
    expected = reference_add_mean(x, other, dim=1)
    actual = candidate.add_mean(x, other, dim=1)
    _assert_close(actual, expected)

    x = torch.tensor([1.0, 3.0, 5.0, 7.0], device="cuda")
    expected = reference_add_mean(x, 2.0, alpha=0.25)
    actual = candidate.add_mean(x, 2.0, alpha=0.25)
    _assert_close(actual, expected)
