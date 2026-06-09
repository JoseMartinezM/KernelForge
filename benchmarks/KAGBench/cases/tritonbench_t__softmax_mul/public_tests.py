import torch
from pytorch_reference import softmax_mul as reference_softmax_mul


def _assert_close(actual, expected, *, rtol=1e-5, atol=1e-6):
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (actual, expected)


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    x = torch.tensor([[1.0, 2.0, 0.0], [3.0, -1.0, 4.0]], device="cuda")
    other = torch.tensor([[0.5, 1.0, 1.5], [2.0, 0.25, -0.5]], device="cuda")
    expected = reference_softmax_mul(x, other, dim=1)
    actual = candidate.softmax_mul(x, other, dim=1)
    _assert_close(actual, expected)

    x = torch.tensor([[0.0, 1.0], [2.0, -2.0], [3.0, 0.5]], device="cuda")
    expected = reference_softmax_mul(x, 0.75, dim=0)
    actual = candidate.softmax_mul(x, 0.75, dim=0)
    _assert_close(actual, expected)
