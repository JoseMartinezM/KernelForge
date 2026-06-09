import torch
from pytorch_reference import fused_layer_norm_relu_linear as reference_fused_layer_norm_relu_linear


def _check(input_tensor, weight, bias, normalized_shape, eps=1e-5, rtol=1e-5, atol=1e-5, **kwargs):
    actual = candidate_fn(input_tensor, weight, bias, normalized_shape, eps=eps, **kwargs)
    expected = reference_fused_layer_norm_relu_linear(input_tensor, weight, bias, normalized_shape, eps=eps, **kwargs)
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol)


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"
    global candidate_fn
    candidate_fn = candidate.fused_layer_norm_relu_linear
    device = "cuda"

    torch.manual_seed(2001)
    input_tensor = torch.randn(3, 2, 7, device=device)
    weight = torch.randn(5, 7, device=device)
    bias = torch.randn(5, device=device)
    _check(input_tensor, weight, bias, 5, eps=1e-5)

    torch.manual_seed(2002)
    input_tensor = torch.randn(1, 9, device=device, dtype=torch.float64)
    weight = torch.randn(6, 9, device=device, dtype=torch.float64)
    bias = torch.randn(6, device=device, dtype=torch.float64)
    _check(input_tensor, weight, bias, (6,), eps=1e-7, rtol=1e-8, atol=1e-8, elementwise_affine=False)

    torch.manual_seed(2003)
    input_tensor = torch.randn(2, 3, 11, device=device)
    weight = torch.randn(8, 11, device=device)
    _check(input_tensor, weight, None, torch.Size([8]), eps=1e-3, rtol=2e-5, atol=2e-5)
