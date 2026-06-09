import torch

from pytorch_reference import fused_hardsigmoid_batch_norm as reference_fused_hardsigmoid_batch_norm


def unit_tests(candidate):
    assert torch.cuda.is_available()
    device = "cuda"

    torch.manual_seed(456)
    x = torch.randn(2, 5, 3, 7, device=device)
    running_mean = torch.linspace(-0.4, 0.4, 5, device=device)
    running_var = torch.linspace(0.35, 1.75, 5, device=device)
    weight = torch.tensor([0.5, -0.75, 1.25, 0.0, 2.0], device=device)
    bias = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.0], device=device)
    actual = candidate.fused_hardsigmoid_batch_norm(x, running_mean, running_var, weight=weight, bias=bias, eps=2e-3)
    expected = reference_fused_hardsigmoid_batch_norm(x, running_mean, running_var, weight=weight, bias=bias, eps=2e-3)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)

    torch.manual_seed(654)
    x64 = torch.randn(3, 2, 4, device=device, dtype=torch.float64)
    running_mean64 = torch.randn(2, device=device, dtype=torch.float64)
    running_var64 = torch.rand(2, device=device, dtype=torch.float64) + 0.4
    weight64 = torch.randn(2, device=device, dtype=torch.float64)
    bias64 = torch.randn(2, device=device, dtype=torch.float64)
    actual64 = candidate.fused_hardsigmoid_batch_norm(x64, running_mean64, running_var64, weight64, bias64, eps=1e-6)
    expected64 = reference_fused_hardsigmoid_batch_norm(x64, running_mean64, running_var64, weight64, bias64, eps=1e-6)
    assert torch.allclose(actual64, expected64, rtol=1e-7, atol=1e-8)

    torch.manual_seed(777)
    x_inplace = torch.randn(4, 3, device=device)
    x_inplace_before = x_inplace.clone()
    running_mean_inplace = torch.randn(3, device=device)
    running_var_inplace = torch.rand(3, device=device) + 0.2
    actual_inplace = candidate.fused_hardsigmoid_batch_norm(x_inplace, running_mean_inplace, running_var_inplace, inplace=True)
    expected_inplace = reference_fused_hardsigmoid_batch_norm(x_inplace_before.clone(), running_mean_inplace, running_var_inplace, inplace=True)
    assert torch.allclose(actual_inplace, expected_inplace, rtol=1e-5, atol=1e-6)
    assert torch.allclose(x_inplace, x_inplace_before, rtol=0.0, atol=0.0)
