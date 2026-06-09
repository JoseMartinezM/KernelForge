import torch
from pytorch_reference import fused_add_mul_groupnorm as reference_fused_add_mul_groupnorm


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"
    device = "cuda"

    torch.manual_seed(3001)
    input1 = torch.randn(2, 4, 4, 4, device=device)
    input2 = torch.randn(2, 4, 4, 4, device=device)
    weight = torch.randn(4, device=device)
    bias = torch.randn(4, device=device)
    actual = candidate.fused_add_mul_groupnorm(input1, input2, weight, bias, 2)
    expected = reference_fused_add_mul_groupnorm(input1, input2, weight, bias, 2)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)

    torch.manual_seed(3002)
    input1 = torch.randn(3, 6, 5, 7, device=device)
    input2 = torch.randn(1, 6, 1, 1, device=device)
    weight = torch.randn(6, device=device)
    bias = torch.randn(6, device=device)
    actual = candidate.fused_add_mul_groupnorm(input1, input2, weight, bias, 3, eps=1e-4)
    expected = reference_fused_add_mul_groupnorm(input1, input2, weight, bias, 3, eps=1e-4)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
