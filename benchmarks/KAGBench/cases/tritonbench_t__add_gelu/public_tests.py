import torch

from pytorch_reference import add_gelu as reference_add_gelu


def public_tests(candidate):
    assert torch.cuda.is_available()
    device = "cuda"

    x = torch.tensor([-2.0, -0.5, 0.0, 1.5, 3.0], device=device)
    other = torch.tensor([0.25, -1.0, 2.0, 0.5, -0.75], device=device)
    actual = candidate.add_gelu(x, other)
    expected = reference_add_gelu(x, other)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)

    x2 = torch.randn(4, 7, device=device)
    actual2 = candidate.add_gelu(x2, 0.5, alpha=1.75, approximate="tanh")
    expected2 = reference_add_gelu(x2, 0.5, alpha=1.75, approximate="tanh")
    assert torch.allclose(actual2, expected2, rtol=1e-5, atol=1e-6)
