import torch

from pytorch_reference import cos as reference_cos


def _assert_close(actual, expected, *, rtol=1e-4, atol=1e-4):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol)


def _check(candidate, tensor, *, rtol=1e-4, atol=1e-4):
    original = tensor.clone()
    actual = candidate.cos(tensor)
    expected = reference_cos(tensor)
    _assert_close(actual, expected, rtol=rtol, atol=atol)
    assert torch.equal(tensor, original), "candidate.cos must not modify its input"


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    torch.manual_seed(201)
    odd_1d = torch.randn(37, device="cuda", dtype=torch.float32) * 7.0
    _check(candidate, odd_1d)

    torch.manual_seed(202)
    matrix = torch.randn((7, 13), device="cuda", dtype=torch.float32) * 10.0 - 1.5
    _check(candidate, matrix)

    torch.manual_seed(203)
    volume = torch.randn((3, 5, 9), device="cuda", dtype=torch.float32) * 4.0 + 0.25
    _check(candidate, volume)

    torch.manual_seed(204)
    mixed_signs = torch.linspace(-9.0, 9.0, steps=257, device="cuda", dtype=torch.float32)
    _check(candidate, mixed_signs)

    if torch.cuda.get_device_capability()[0] >= 5:
        torch.manual_seed(205)
        half_values = (torch.randn((11, 17), device="cuda", dtype=torch.float16) * 5.0).contiguous()
        _check(candidate, half_values, rtol=1e-3, atol=1e-3)
