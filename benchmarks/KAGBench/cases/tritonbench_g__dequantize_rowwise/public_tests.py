import torch

from pytorch_reference import dequantize_rowwise as reference_dequantize_rowwise


def _check(candidate, x: torch.Tensor, state_x: torch.Tensor) -> None:
    actual = candidate.dequantize_rowwise(x, state_x)
    expected = reference_dequantize_rowwise(x, state_x)

    assert actual.shape == expected.shape
    assert actual.dtype == torch.float16
    assert actual.device == x.device
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def public_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    x = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        device="cuda",
        dtype=torch.int8,
    )
    state_x = torch.tensor([4.0, 8.0], device="cuda", dtype=torch.float32)
    _check(candidate, x, state_x)

    torch.manual_seed(101)
    x = torch.randint(0, 127, (8, 16), device="cuda", dtype=torch.int8)
    state_x = torch.rand(8, device="cuda", dtype=torch.float32) * 10.0
    _check(candidate, x, state_x)
