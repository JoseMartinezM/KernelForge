import torch

from pytorch_reference import cos as reference_cos


def _assert_close(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    torch.manual_seed(101)
    x = torch.rand(1024, device="cuda", dtype=torch.float32) * 6.0 - 3.0
    _assert_close(candidate.cos(x), reference_cos(x))

    torch.manual_seed(102)
    y = torch.rand((32, 64), device="cuda", dtype=torch.float32) * 12.0 - 6.0
    _assert_close(candidate.cos(y), reference_cos(y))
