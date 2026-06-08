import torch

from pytorch_reference import puzzle1 as reference_puzzle1


def _check(candidate, x: torch.Tensor) -> None:
    original = x.clone()
    actual = candidate.puzzle1(x)
    expected = reference_puzzle1(x)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)
    assert torch.allclose(x, original, rtol=0, atol=0)


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    # Includes zero and negative values.
    x_values = torch.tensor([0.0, -1.0, 2.5, -7.25, 13.0], device="cuda", dtype=torch.float32)
    _check(candidate, x_values)

    # Odd, non-power-of-two number of elements.
    torch.manual_seed(201)
    x_odd = torch.randn(37, device="cuda", dtype=torch.float32)
    _check(candidate, x_odd)

    # Float16 coverage with a non-power-of-two 2D shape.
    torch.manual_seed(202)
    x_half = torch.randn(7, 9, device="cuda", dtype=torch.float16)
    _check(candidate, x_half)

    # Larger odd shape crossing typical implementation boundaries.
    torch.manual_seed(203)
    x_large = torch.randn(1031, device="cuda", dtype=torch.float32)
    _check(candidate, x_large)

    # Non-contiguous input should still behave like elementwise x + 10.
    torch.manual_seed(204)
    base = torch.randn(6, 11, device="cuda", dtype=torch.float32)
    x_noncontiguous = base.t()
    assert not x_noncontiguous.is_contiguous()
    _check(candidate, x_noncontiguous)
