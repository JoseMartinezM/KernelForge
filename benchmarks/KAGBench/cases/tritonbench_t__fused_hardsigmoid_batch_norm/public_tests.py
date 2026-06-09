import torch

from pytorch_reference import fused_hardsigmoid_batch_norm as reference_fused_hardsigmoid_batch_norm


def public_tests(candidate):
    assert torch.cuda.is_available()
    torch.manual_seed(10)
    device = "cuda"

    x = torch.randn(4, 3, 6, 5, device=device)
    running_mean = torch.tensor([0.0, 0.25, -0.5], device=device)
    running_var = torch.tensor([1.0, 0.7, 1.8], device=device)
    actual = candidate.fused_hardsigmoid_batch_norm(x, running_mean, running_var)
    expected = reference_fused_hardsigmoid_batch_norm(x, running_mean, running_var)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)

    x2 = torch.randn(7, 4, device=device)
    running_mean2 = torch.randn(4, device=device) * 0.2
    running_var2 = torch.rand(4, device=device) + 0.5
    weight2 = torch.randn(4, device=device) * 0.5 + 1.0
    bias2 = torch.randn(4, device=device) * 0.1
    actual2 = candidate.fused_hardsigmoid_batch_norm(x2, running_mean2, running_var2, weight2, bias2, eps=5e-4)
    expected2 = reference_fused_hardsigmoid_batch_norm(x2, running_mean2, running_var2, weight2, bias2, eps=5e-4)
    assert torch.allclose(actual2, expected2, rtol=1e-5, atol=1e-6)
