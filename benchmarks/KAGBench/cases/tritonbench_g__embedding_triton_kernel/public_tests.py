import torch

from pytorch_reference import embedding as reference_embedding


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1e-6, 1e-6


def _check(candidate, input_ids: torch.Tensor, weight: torch.Tensor, vob_start_id: int, vob_end_id: int) -> None:
    actual = torch.empty((input_ids.numel(), weight.shape[1]), device=weight.device, dtype=weight.dtype)
    expected = torch.empty_like(actual)
    actual.fill_(-7.0)
    expected.fill_(-7.0)

    candidate.embedding(input_ids, weight, vob_start_id, vob_end_id, actual)
    reference_embedding(input_ids, weight, vob_start_id, vob_end_id, expected)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    rtol, atol = _tolerances(weight.dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    torch.manual_seed(101)
    weight = torch.randn(16, 4, device="cuda", dtype=torch.float32)
    input_ids = torch.tensor([0, 3, 5, 7, 11, 15], device="cuda", dtype=torch.int64)
    _check(candidate, input_ids, weight, 0, 16)

    torch.manual_seed(102)
    weight = torch.randn(32, 8, device="cuda", dtype=torch.float32)
    input_ids = torch.randint(0, 32, (24,), device="cuda", dtype=torch.int32)
    _check(candidate, input_ids, weight, 0, 32)