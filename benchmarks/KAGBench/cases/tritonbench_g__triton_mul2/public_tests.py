import torch

from pytorch_reference import triton_mul2 as reference_triton_mul2
from pytorch_reference import triton_mul2_inplace as reference_triton_mul2_inplace


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1e-6, 1e-6


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    rtol, atol = _tolerances(expected.dtype)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol)


def _check_out_of_place(candidate, x: torch.Tensor, block_size: int) -> None:
    original = x.clone()
    actual = candidate.triton_mul2(x, BLOCK_SIZE=block_size)
    expected = reference_triton_mul2(original, BLOCK_SIZE=block_size)

    _assert_close(actual, expected)
    assert torch.allclose(x, original, rtol=0, atol=0)
    assert actual.data_ptr() != x.data_ptr()


def _check_inplace(candidate, x: torch.Tensor, block_size: int) -> None:
    expected_input = x.clone()
    expected = reference_triton_mul2_inplace(expected_input, BLOCK_SIZE=block_size)
    original_ptr = x.data_ptr()

    actual = candidate.triton_mul2_inplace(x, BLOCK_SIZE=block_size)

    assert x.data_ptr() == original_ptr
    assert actual.data_ptr() == original_ptr
    _assert_close(x, expected)
    _assert_close(actual, expected)


def _check_both(candidate, x: torch.Tensor, block_size: int) -> None:
    _check_out_of_place(candidate, x, block_size)
    _check_inplace(candidate, x.clone(), block_size)


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    torch.manual_seed(101)
    _check_both(candidate, torch.randn(17, device="cuda", dtype=torch.float32), 16)

    torch.manual_seed(102)
    _check_both(candidate, torch.randn(3, 5, device="cuda", dtype=torch.float16), 8)
