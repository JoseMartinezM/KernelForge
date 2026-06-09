import torch

from pytorch_reference import sigmoid_batch_norm as reference_sigmoid_batch_norm


def unit_tests(candidate):
    assert torch.cuda.is_available()
    device = "cuda"

    torch.manual_seed(123)
    x = torch.randn(2, 3, 5, 7, device=device)
    running_mean = torch.tensor([-0.4, 0.0, 0.25], device=device)
    running_var = torch.tensor([0.6, 1.2, 2.5], device=device)
    weight = torch.tensor([0.75, -1.25, 1.5], device=device)
    bias = torch.tensor([0.2, -0.1, 0.05], device=device)
    actual = candidate.sigmoid_batch_norm(x, running_mean, running_var, weight=weight, bias=bias, eps=7e-4)
    expected = reference_sigmoid_batch_norm(x, running_mean, running_var, weight=weight, bias=bias, eps=7e-4)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)

    torch.manual_seed(321)
    x64 = torch.randn(5, 2, device=device, dtype=torch.float64)
    running_mean64 = torch.randn(2, device=device, dtype=torch.float64)
    running_var64 = torch.rand(2, device=device, dtype=torch.float64) + 0.25
    weight64 = torch.randn(2, device=device, dtype=torch.float64)
    bias64 = torch.randn(2, device=device, dtype=torch.float64)
    actual64 = candidate.sigmoid_batch_norm(x64, running_mean64, running_var64, weight64, bias64, eps=1e-6)
    expected64 = reference_sigmoid_batch_norm(x64, running_mean64, running_var64, weight64, bias64, eps=1e-6)
    assert torch.allclose(actual64, expected64, rtol=1e-7, atol=1e-8)

    torch.manual_seed(999)
    x_no_affine = torch.randn(1, 4, 3, device=device)
    running_mean_no_affine = torch.randn(4, device=device)
    running_var_no_affine = torch.rand(4, device=device) + 0.1
    actual_no_affine = candidate.sigmoid_batch_norm(x_no_affine, running_mean_no_affine, running_var_no_affine)
    expected_no_affine = reference_sigmoid_batch_norm(x_no_affine, running_mean_no_affine, running_var_no_affine)
    assert torch.allclose(actual_no_affine, expected_no_affine, rtol=1e-5, atol=1e-6)
