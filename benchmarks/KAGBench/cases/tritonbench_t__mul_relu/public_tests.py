import torch

from pytorch_reference import mul_relu as reference_mul_relu


def public_tests(candidate):
    assert torch.cuda.is_available()
    device = "cuda"

    x = torch.tensor([-3.0, -1.0, 0.0, 2.0, 4.0], device=device)
    other = torch.tensor([2.0, -5.0, 7.0, -1.5, 0.25], device=device)
    actual = candidate.mul_relu(x, other)
    expected = reference_mul_relu(x, other)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)

    x2 = torch.tensor([[-2.0, 1.5, 3.0], [0.5, -4.0, 2.0]], device=device)
    actual2 = candidate.mul_relu(x2, -0.5)
    expected2 = reference_mul_relu(x2, -0.5)
    assert torch.allclose(actual2, expected2, rtol=1e-6, atol=1e-6)
