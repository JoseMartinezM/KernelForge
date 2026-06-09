import torch
from pytorch_reference import fused_layer_norm_relu_linear as reference_fused_layer_norm_relu_linear


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"
    device = "cuda"

    torch.manual_seed(1001)
    input_tensor = torch.randn(4, 5, device=device)
    weight = torch.randn(3, 5, device=device)
    bias = torch.randn(3, device=device)
    actual = candidate.fused_layer_norm_relu_linear(input_tensor, weight, bias, 3)
    expected = reference_fused_layer_norm_relu_linear(input_tensor, weight, bias, 3)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)

    torch.manual_seed(1002)
    input_tensor = torch.randn(2, 7, device=device)
    weight = torch.randn(4, 7, device=device)
    actual = candidate.fused_layer_norm_relu_linear(
        input_tensor,
        weight,
        None,
        torch.Size([4]),
        eps=1e-4,
    )
    expected = reference_fused_layer_norm_relu_linear(
        input_tensor,
        weight,
        None,
        torch.Size([4]),
        eps=1e-4,
    )
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
