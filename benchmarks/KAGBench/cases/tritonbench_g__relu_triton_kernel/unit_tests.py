import torch

from pytorch_reference import relu as reference_relu


def _check(candidate, x: torch.Tensor, *, rtol: float = 1e-5, atol: float = 1e-6) -> None:
    actual = candidate.relu(x)
    expected = reference_relu(x)
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    assert actual.device.type == "cuda"
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol)


def unit_tests(candidate):
    # Exact negative, zero, and positive values.
    x = torch.tensor(
        [-7.0, -1.5, -0.0, 0.0, 0.25, 3.5, 11.0],
        device="cuda",
        dtype=torch.float32,
    )
    _check(candidate, x)

    # Odd size larger than a common single-block boundary.
    torch.manual_seed(201)
    x = torch.randn(1031, device="cuda", dtype=torch.float32) * 4.0 - 1.0
    x[0] = -5.0
    x[17] = 0.0
    x[-1] = 6.0
    _check(candidate, x)

    # Higher-rank float32 tensor.
    torch.manual_seed(202)
    x = torch.randn(3, 5, 7, device="cuda", dtype=torch.float32) * 2.0
    x[0, 0, 0] = -2.0
    x[1, 2, 3] = 0.0
    x[2, 4, 6] = 2.0
    _check(candidate, x)

    # Float16 inputs should still produce float32 outputs.
    torch.manual_seed(203)
    x = (torch.randn(257, device="cuda", dtype=torch.float16) * 3.0).to(torch.float16)
    x[0] = torch.tensor(-4.0, device="cuda", dtype=torch.float16)
    x[1] = torch.tensor(0.0, device="cuda", dtype=torch.float16)
    x[-1] = torch.tensor(4.0, device="cuda", dtype=torch.float16)
    _check(candidate, x, rtol=1e-3, atol=1e-3)
