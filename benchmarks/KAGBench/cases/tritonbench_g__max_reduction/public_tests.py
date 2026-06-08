import torch

from pytorch_reference import max as reference_max
from pytorch_reference import max_dim as reference_max_dim


def _unique_values(shape, dtype: torch.dtype, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    total = 1
    for size in shape:
        total *= size
    offsets = torch.arange(total, device="cuda", dtype=torch.float32).reshape(shape) * 0.125
    jitter = torch.rand(shape, device="cuda", dtype=torch.float32) * 0.03125
    return (offsets + jitter - 4.0).to(dtype).contiguous()


def _as_values_indices(result):
    if hasattr(result, "values") and hasattr(result, "indices"):
        return result.values, result.indices
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    raise AssertionError("max_dim must return values and indices")


def _assert_same_tensor(actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 0.0, rtol: float = 0.0) -> None:
    assert isinstance(actual, torch.Tensor), f"expected tensor output, got {type(actual).__name__}"
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"dtype mismatch: {actual.dtype} != {expected.dtype}"
    assert actual.device == expected.device, f"device mismatch: {actual.device} != {expected.device}"
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _check_max(candidate, inp: torch.Tensor, *, atol: float = 0.0, rtol: float = 0.0) -> None:
    actual = candidate.max(inp)
    expected = reference_max(inp)
    _assert_same_tensor(actual, expected, atol=atol, rtol=rtol)


def _check_max_dim(candidate, inp: torch.Tensor, dim: int, keepdim: bool = False, *, atol: float = 0.0, rtol: float = 0.0) -> None:
    actual_values, actual_indices = _as_values_indices(candidate.max_dim(inp, dim=dim, keepdim=keepdim))
    expected = reference_max_dim(inp, dim=dim, keepdim=keepdim)
    _assert_same_tensor(actual_values, expected.values, atol=atol, rtol=rtol)
    _assert_same_tensor(actual_indices, expected.indices)


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    inp = _unique_values((64,), torch.float32, seed=101)
    _check_max(candidate, inp)

    inp = _unique_values((4, 7), torch.float32, seed=102)
    _check_max_dim(candidate, inp, dim=1)

    inp = _unique_values((3, 5, 6), torch.float32, seed=103)
    _check_max_dim(candidate, inp, dim=2, keepdim=True)
