import torch

from pytorch_reference import logsumexp_fwd as reference_logsumexp_fwd


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float = 1e-4, atol: float = 1e-4) -> None:
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"dtype mismatch: {actual.dtype} != {expected.dtype}"
    assert actual.device == expected.device, f"device mismatch: {actual.device} != {expected.device}"
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _check(candidate, x: torch.Tensor, *, rtol: float = 1e-4, atol: float = 1e-4, **kwargs) -> None:
    assert x.is_contiguous()
    original = x.clone()
    actual = candidate.logsumexp_fwd(x, **kwargs)
    expected = reference_logsumexp_fwd(x, **kwargs)
    _assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(x, original, rtol=0.0, atol=0.0)


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    # Odd last dimension.
    torch.manual_seed(201)
    odd_last_dim = torch.randn(5, 7, device="cuda", dtype=torch.float32)
    _check(candidate, odd_last_dim)

    # Higher-rank shape: reduce only the last dimension.
    torch.manual_seed(202)
    higher_rank = torch.randn(2, 3, 4, 9, device="cuda", dtype=torch.float32)
    _check(candidate, higher_rank)

    # Explicit scale before reduction.
    torch.manual_seed(203)
    scaled = torch.randn(4, 11, device="cuda", dtype=torch.float32) * 2.0 - 1.0
    _check(candidate, scaled, scale=1.75)

    # dtype requests other than torch.float/torch.float32 cast only the final result.
    torch.manual_seed(204)
    double_out = torch.randn(3, 13, device="cuda", dtype=torch.float32)
    _check(candidate, double_out, dtype=torch.float64)

    # Large positive and negative values should remain numerically stable.
    stable_values = torch.tensor(
        [
            [-1000.0, -999.0, -1001.0, -1005.0, -998.0],
            [1000.0, 1001.0, 999.0, 995.0, 1002.0],
            [-80.0, -20.0, 0.0, 20.0, 80.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    _check(candidate, stable_values)

    # Float16 inputs still produce float32 by default; allow lower-precision tolerance.
    torch.manual_seed(205)
    half_input = (torch.randn(3, 5, 17, device="cuda", dtype=torch.float16) * 3.0).contiguous()
    _check(candidate, half_input, rtol=3e-3, atol=3e-3)

    # dtype can also request a lower-precision final cast.
    torch.manual_seed(206)
    half_out = torch.randn(2, 15, device="cuda", dtype=torch.float32)
    _check(candidate, half_out, dtype=torch.float16, rtol=3e-3, atol=3e-3)
