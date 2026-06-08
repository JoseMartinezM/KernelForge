import torch

from pytorch_reference import dequantize_rowwise as reference_dequantize_rowwise


def _check(candidate, x: torch.Tensor, state_x: torch.Tensor, *, rtol: float = 1e-3, atol: float = 1e-3) -> None:
    original_x = x.clone()
    original_state_x = state_x.clone()

    actual = candidate.dequantize_rowwise(x, state_x)
    expected = reference_dequantize_rowwise(x, state_x)

    assert actual.shape == x.shape
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float16
    assert actual.device == x.device
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(state_x, original_state_x, rtol=0, atol=0)


def unit_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    # Non-power-of-two row and column counts with mixed signs.
    x = torch.tensor(
        [
            [-128, -64, -1, 0, 1],
            [2, 7, -8, 31, -32],
            [63, -96, 100, -120, 126],
        ],
        device="cuda",
        dtype=torch.int8,
    )
    state_x = torch.tensor([1.0, 2.5, 7.0], device="cuda", dtype=torch.float32)
    _check(candidate, x, state_x)

    # Zero scale rows should produce exactly zero for the whole row.
    torch.manual_seed(201)
    x = torch.randint(-128, 127, (5, 7), device="cuda", dtype=torch.int8)
    state_x = torch.tensor([0.0, 3.0, 0.0, 1.25, 0.5], device="cuda", dtype=torch.float32)
    _check(candidate, x, state_x)

    # Float16 scale input with odd dimensions.
    torch.manual_seed(202)
    x = torch.randint(-128, 127, (7, 13), device="cuda", dtype=torch.int8)
    state_x = (torch.rand(7, device="cuda", dtype=torch.float16) * 6.0).to(torch.float16)
    state_x[0] = torch.tensor(0.0, device="cuda", dtype=torch.float16)
    _check(candidate, x, state_x, rtol=2e-3, atol=2e-3)

    # Float32 scale input with more rows than columns.
    torch.manual_seed(203)
    x = torch.randint(-128, 127, (9, 3), device="cuda", dtype=torch.int8)
    state_x = torch.linspace(0.5, 4.0, steps=9, device="cuda", dtype=torch.float32)
    _check(candidate, x, state_x)

    # Wider non-power-of-two column count with deterministic random values.
    torch.manual_seed(204)
    x = torch.randint(-128, 127, (4, 33), device="cuda", dtype=torch.int8)
    state_x = torch.rand(4, device="cuda", dtype=torch.float32) * 12.0
    _check(candidate, x, state_x)
