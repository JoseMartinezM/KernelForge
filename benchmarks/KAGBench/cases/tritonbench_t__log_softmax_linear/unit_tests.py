import torch

from pytorch_reference import log_softmax_linear as reference_log_softmax_linear


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"
    device = "cuda"

    torch.manual_seed(1234)
    input_3d = torch.randn(2, 5, 7, device=device, dtype=torch.float32)
    weight = torch.randn(11, 7, device=device, dtype=torch.float32)
    bias = torch.randn(11, device=device, dtype=torch.float32)
    expected = reference_log_softmax_linear(input_3d, weight, bias, dim=-1)
    actual = candidate.log_softmax_linear(input_3d, weight, bias, dim=-1)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)

    torch.manual_seed(2025)
    input_dim0 = torch.randn(7, 3, device=device, dtype=torch.float32)
    weight_dim0 = torch.randn(5, 3, device=device, dtype=torch.float32)
    bias_dim0 = torch.randn(5, device=device, dtype=torch.float32)
    expected_dim0 = reference_log_softmax_linear(input_dim0, weight_dim0, bias_dim0, dim=0)
    actual_dim0 = candidate.log_softmax_linear(input_dim0, weight_dim0, bias_dim0, dim=0)
    assert torch.allclose(actual_dim0, expected_dim0, rtol=1e-5, atol=1e-6)

    torch.manual_seed(31415)
    input_no_bias = torch.randn(9, 4, device=device, dtype=torch.float32)
    weight_no_bias = torch.randn(6, 4, device=device, dtype=torch.float32)
    expected_no_bias = reference_log_softmax_linear(input_no_bias, weight_no_bias, dtype=torch.float64)
    actual_no_bias = candidate.log_softmax_linear(input_no_bias, weight_no_bias, dtype=torch.float64)
    assert actual_no_bias.dtype == torch.float64
    assert torch.allclose(actual_no_bias, expected_no_bias, rtol=1e-10, atol=1e-12)
