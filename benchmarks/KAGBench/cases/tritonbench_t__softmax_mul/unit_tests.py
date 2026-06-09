import torch
from pytorch_reference import softmax_mul as reference_softmax_mul


def _assert_close(actual, expected, *, rtol=1e-5, atol=1e-6):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (actual, expected)


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"
    torch.manual_seed(2025)

    x = torch.randn((3, 5, 7), device="cuda", dtype=torch.float32)
    other = torch.randn((3, 5, 7), device="cuda", dtype=torch.float32)
    expected = reference_softmax_mul(x, other, dim=-1)
    actual = candidate.softmax_mul(x, other, dim=-1)
    _assert_close(actual, expected)

    x = torch.randn((5, 3), device="cuda", dtype=torch.float16)
    expected = reference_softmax_mul(x, 1.5, dim=0, dtype=torch.float32)
    actual = candidate.softmax_mul(x, 1.5, dim=0, dtype=torch.float32)
    _assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    x = torch.randn((2, 7, 3), device="cuda", dtype=torch.float32)
    other = torch.randn((2, 7, 3), device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    expected_out = torch.empty_like(x)
    expected = reference_softmax_mul(x, other, dim=1, out=expected_out)
    actual = candidate.softmax_mul(x, other, dim=1, out=out)
    assert actual is out
    _assert_close(actual, expected)
