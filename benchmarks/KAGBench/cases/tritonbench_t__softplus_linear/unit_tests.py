import torch

from pytorch_reference import softplus_linear as reference_softplus_linear


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"
    device = "cuda"

    torch.manual_seed(4321)
    input_odd = torch.randn(5, 7, device=device, dtype=torch.float32)
    weight_odd = torch.randn(3, 7, device=device, dtype=torch.float32)
    bias_odd = torch.randn(3, device=device, dtype=torch.float32)
    expected_odd = reference_softplus_linear(input_odd, weight_odd, bias_odd, beta=0.5, threshold=4.0)
    actual_odd = candidate.softplus_linear(input_odd, weight_odd, bias_odd, beta=0.5, threshold=4.0)
    assert torch.allclose(actual_odd, expected_odd, rtol=1e-5, atol=1e-6)

    torch.manual_seed(9876)
    input_no_bias = torch.randn(2, 3, 5, device=device, dtype=torch.float32)
    weight_no_bias = torch.randn(11, 5, device=device, dtype=torch.float32)
    expected_no_bias = reference_softplus_linear(input_no_bias, weight_no_bias, beta=2.5, threshold=6.0)
    actual_no_bias = candidate.softplus_linear(input_no_bias, weight_no_bias, beta=2.5, threshold=6.0)
    assert torch.allclose(actual_no_bias, expected_no_bias, rtol=1e-5, atol=1e-6)

    input_large = torch.tensor([[20.0, -5.0, 3.0], [-18.0, 2.0, 1.0]], device=device, dtype=torch.float64)
    weight_large = torch.tensor([[1.0, 0.5, -0.25], [-0.75, 1.5, 0.5]], device=device, dtype=torch.float64)
    bias_large = torch.tensor([0.25, -1.0], device=device, dtype=torch.float64)
    expected_large = reference_softplus_linear(input_large, weight_large, bias_large, beta=3.0, threshold=10.0)
    actual_large = candidate.softplus_linear(input_large, weight_large, bias_large, beta=3.0, threshold=10.0)
    assert torch.allclose(actual_large, expected_large, rtol=1e-12, atol=1e-12)
