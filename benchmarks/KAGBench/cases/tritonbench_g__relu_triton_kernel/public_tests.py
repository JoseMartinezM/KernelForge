import torch

from pytorch_reference import relu as reference_relu


def _check(candidate, x: torch.Tensor) -> None:
    actual = candidate.relu(x)
    expected = reference_relu(x)
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    assert actual.device.type == "cuda"
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def public_tests(candidate):
    torch.manual_seed(101)
    x = torch.randn(128, device="cuda", dtype=torch.float32) * 3.0
    _check(candidate, x)

    torch.manual_seed(102)
    x = torch.randn(4, 17, device="cuda", dtype=torch.float32) - 0.25
    _check(candidate, x)
