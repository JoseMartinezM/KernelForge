import torch

from pytorch_reference import add as reference_add

def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float16:
        return 1e-2, 1e-2
    return 1e-5, 1e-5


def _check(candidate, size: int, dtype: torch.dtype, seed: int) -> None:
    torch.manual_seed(seed)
    x = torch.randn(size, device="cuda", dtype=dtype)
    y = torch.randn(size, device="cuda", dtype=dtype)
    actual = candidate.add(x, y)
    expected = reference_add(x, y)
    rtol, atol = _tolerances(dtype)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (size, dtype, actual, expected)


def unit_tests(candidate):
    # The source API indexes tensors linearly and is intended for contiguous inputs.
    # Non-contiguous tensors are therefore not included in the scoring cases.
    _check(candidate, size=1, dtype=torch.float32, seed=10)
    _check(candidate, size=17, dtype=torch.float32, seed=11)
    _check(candidate, size=513, dtype=torch.float32, seed=12)
    _check(candidate, size=98433, dtype=torch.float32, seed=13)
    _check(candidate, size=1_000_003, dtype=torch.float32, seed=14)
    _check(candidate, size=129, dtype=torch.float16, seed=15)
    _check(candidate, size=65537, dtype=torch.float16, seed=16)
