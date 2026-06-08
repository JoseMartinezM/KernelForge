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


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    values = torch.tensor([0.0, -1.5, 2.25, -3.75, 8.0], device="cuda", dtype=torch.float32)
    _check_both(candidate, values, 16)

    torch.manual_seed(201)
    x_odd = torch.randn(37, device="cuda", dtype=torch.float32)
    _check_both(candidate, x_odd, 8)

    torch.manual_seed(202)
    x_many_blocks = torch.randn(257, device="cuda", dtype=torch.float32)
    for block_size in (1, 16, 64):
        _check_both(candidate, x_many_blocks.clone(), block_size)

    torch.manual_seed(203)
    x_high_rank = torch.randn(3, 5, 7, device="cuda", dtype=torch.float32).contiguous()
    assert x_high_rank.is_contiguous()
    _check_both(candidate, x_high_rank, 32)

    torch.manual_seed(204)
    x_half = torch.randn(11, 13, device="cuda", dtype=torch.float16)
    _check_both(candidate, x_half, 16)

    torch.manual_seed(205)
    x_half_3d = torch.randn(2, 7, 9, device="cuda", dtype=torch.float16).contiguous()
    assert x_half_3d.is_contiguous()
    _check_both(candidate, x_half_3d, 64)
