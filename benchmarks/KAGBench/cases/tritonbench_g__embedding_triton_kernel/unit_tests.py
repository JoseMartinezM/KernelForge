import torch

from pytorch_reference import embedding as reference_embedding


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1e-6, 1e-6


def _check(candidate, input_ids: torch.Tensor, weight: torch.Tensor, vob_start_id: int, vob_end_id: int) -> None:
    original_input_ids = input_ids.clone()
    original_weight = weight.clone()
    actual = torch.empty((input_ids.numel(), weight.shape[1]), device=weight.device, dtype=weight.dtype)
    expected = torch.empty_like(actual)
    actual.fill_(-7.0)
    expected.fill_(-7.0)

    candidate.embedding(input_ids, weight, vob_start_id, vob_end_id, actual)
    reference_embedding(input_ids, weight, vob_start_id, vob_end_id, expected)

    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"dtype mismatch: {actual.dtype} != {expected.dtype}"
    assert actual.device == expected.device, f"device mismatch: {actual.device} != {expected.device}"
    rtol, atol = _tolerances(weight.dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(input_ids, original_input_ids, rtol=0, atol=0)
    torch.testing.assert_close(weight, original_weight, rtol=0, atol=0)


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    # Nonzero vocabulary start offset with all ids in range.
    torch.manual_seed(201)
    weight = torch.randn(20, 7, device="cuda", dtype=torch.float32)
    input_ids = torch.tensor([100, 101, 115, 104, 110, 100, 119], device="cuda", dtype=torch.int64)
    _check(candidate, input_ids, weight, 100, 120)

    # Rows for ids below the start, at the end, or above the end must be zeroed.
    torch.manual_seed(202)
    weight = torch.randn(8, 5, device="cuda", dtype=torch.float32)
    input_ids = torch.tensor([-3, 0, 1, 7, 8, 99, 4, 3, 2], device="cuda", dtype=torch.int32)
    _check(candidate, input_ids, weight, 0, 8)

    # Odd sequence length and non-power-of-two embedding dimension.
    torch.manual_seed(203)
    weight = torch.randn(32, 33, device="cuda", dtype=torch.float32)
    input_ids = torch.randint(0, 32, (17,), device="cuda", dtype=torch.int32)
    _check(candidate, input_ids, weight, 0, 32)

    # Float16 data with nonzero offset and mixed valid/invalid ids.
    torch.manual_seed(204)
    weight = torch.randn(24, 11, device="cuda", dtype=torch.float16)
    input_ids = torch.tensor([4, 5, 6, 17, 28, 29, 12, 5, 31, 20, 7], device="cuda", dtype=torch.int64)
    _check(candidate, input_ids, weight, 5, 29)

    # Larger valid range with a different embedding dimension and an odd token count.
    torch.manual_seed(205)
    weight = torch.randn(64, 1, device="cuda", dtype=torch.float32)
    input_ids = torch.randint(50, 114, (65,), device="cuda", dtype=torch.int64)
    _check(candidate, input_ids, weight, 50, 114)