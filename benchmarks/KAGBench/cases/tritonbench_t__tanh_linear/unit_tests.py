import torch
from pytorch_reference import tanh_linear as reference_tanh_linear


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required"

    torch.manual_seed(123)
    input_3d = torch.randn(2, 3, 5, device="cuda", dtype=torch.float32)
    weight_3d = torch.randn(7, 5, device="cuda", dtype=torch.float32)
    bias_3d = torch.randn(7, device="cuda", dtype=torch.float32)
    expected_3d = reference_tanh_linear(input_3d, weight_3d, bias_3d)
    actual_3d = candidate.tanh_linear(input_3d, weight_3d, bias_3d)
    assert torch.allclose(actual_3d, expected_3d, rtol=1e-5, atol=1e-5)

    torch.manual_seed(456)
    input_odd = torch.randn(11, 13, device="cuda", dtype=torch.float32)
    weight_odd = torch.randn(9, 13, device="cuda", dtype=torch.float32)
    expected_odd = reference_tanh_linear(input_odd, weight_odd, None)
    actual_odd = candidate.tanh_linear(input_odd, weight_odd, None)
    assert torch.allclose(actual_odd, expected_odd, rtol=1e-5, atol=1e-5)

    torch.manual_seed(789)
    input_half = torch.randn(4, 17, device="cuda", dtype=torch.float16)
    weight_half = torch.randn(6, 17, device="cuda", dtype=torch.float16)
    bias_half = torch.randn(6, device="cuda", dtype=torch.float16)
    expected_half = reference_tanh_linear(input_half, weight_half, bias_half)
    actual_half = candidate.tanh_linear(input_half, weight_half, bias_half)
    assert torch.allclose(actual_half, expected_half, rtol=2e-3, atol=2e-3)
