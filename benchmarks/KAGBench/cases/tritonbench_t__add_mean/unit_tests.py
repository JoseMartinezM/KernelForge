import torch
from pytorch_reference import add_mean as reference_add_mean


def _assert_close(actual, expected, *, rtol=1e-5, atol=1e-6):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (actual, expected)


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"
    torch.manual_seed(1234)

    x = torch.randn((3, 5, 7), device="cuda", dtype=torch.float32)
    other = torch.randn((3, 5, 7), device="cuda", dtype=torch.float32)
    expected = reference_add_mean(x, other, dim=(0, 2), alpha=-0.75, keepdim=True)
    actual = candidate.add_mean(x, other, dim=(0, 2), alpha=-0.75, keepdim=True)
    _assert_close(actual, expected)

    x = torch.randn((5, 3), device="cuda", dtype=torch.float16)
    expected = reference_add_mean(x, 1.25, dim=0, dtype=torch.float32)
    actual = candidate.add_mean(x, 1.25, dim=0, dtype=torch.float32)
    _assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    x = torch.randn((2, 3, 4), device="cuda", dtype=torch.float32)
    other = torch.randn((2, 3, 4), device="cuda", dtype=torch.float32)
    expected = reference_add_mean(x, other, dim=-1, keepdim=False)
    actual = candidate.add_mean(x, other, dim=-1, keepdim=False)
    _assert_close(actual, expected)
