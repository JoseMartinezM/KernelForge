import torch

from pytorch_reference import sin_triton as reference_sin_triton


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1e-4, 1e-4


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol)


def _check(candidate, x: torch.Tensor, out: torch.Tensor) -> None:
    assert x.is_cuda and out.is_cuda
    assert x.is_contiguous() and out.is_contiguous()
    assert x.shape == out.shape

    original_x = x.clone()
    expected_out = out.clone()
    rtol, atol = _tolerances(x.dtype)

    reference_sin_triton(original_x, expected_out)
    candidate.sin_triton(x, out)

    _assert_close(out, expected_out, rtol=rtol, atol=atol)
    assert torch.equal(x, original_x), "candidate.sin_triton must not modify x"


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    torch.manual_seed(201)
    odd = torch.randn(37, device="cuda", dtype=torch.float32) * 5.0 - 1.5
    _check(candidate, odd, torch.empty_like(odd))

    torch.manual_seed(202)
    non_power_two = torch.randn(1031, device="cuda", dtype=torch.float32) * 3.0
    _check(candidate, non_power_two, torch.full_like(non_power_two, 99.0))

    torch.manual_seed(203)
    higher_rank = torch.randn((3, 5, 7), device="cuda", dtype=torch.float32) * 8.0 - 4.0
    higher_rank_out = torch.full_like(higher_rank, -7.0)
    assert higher_rank.is_contiguous() and higher_rank_out.is_contiguous()
    _check(candidate, higher_rank, higher_rank_out)

    negative_values = torch.linspace(-9.0, -0.25, steps=257, device="cuda", dtype=torch.float32)
    _check(candidate, negative_values, torch.full_like(negative_values, 123.5))

    if torch.cuda.get_device_capability()[0] >= 5:
        torch.manual_seed(204)
        half_values = (torch.randn((11, 17), device="cuda", dtype=torch.float16) * 4.0 - 2.0).contiguous()
        _check(candidate, half_values, torch.full_like(half_values, -3.0))
