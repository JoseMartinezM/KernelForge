import torch
from pytorch_reference import tanh_linear as reference_tanh_linear


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required"
    torch.manual_seed(0)

    input = torch.randn(5, 3, device="cuda", dtype=torch.float32)
    weight = torch.randn(4, 3, device="cuda", dtype=torch.float32)
    bias = torch.randn(4, device="cuda", dtype=torch.float32)
    expected = reference_tanh_linear(input, weight, bias)
    actual = candidate.tanh_linear(input, weight, bias)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)

    input_no_bias = torch.randn(7, 2, device="cuda", dtype=torch.float32)
    weight_no_bias = torch.randn(5, 2, device="cuda", dtype=torch.float32)
    expected_no_bias = reference_tanh_linear(input_no_bias, weight_no_bias)
    actual_no_bias = candidate.tanh_linear(input_no_bias, weight_no_bias)
    assert torch.allclose(actual_no_bias, expected_no_bias, rtol=1e-5, atol=1e-5)
