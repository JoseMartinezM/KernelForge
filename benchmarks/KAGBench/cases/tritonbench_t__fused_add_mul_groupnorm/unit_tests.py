import torch
from pytorch_reference import fused_add_mul_groupnorm as reference_fused_add_mul_groupnorm


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"
    device = "cuda"

    torch.manual_seed(4001)
    input1 = torch.randn(1, 8, 3, 5, device=device)
    input2 = torch.randn(1, 8, 3, 5, device=device)
    weight = torch.randn(8, device=device)
    bias = torch.randn(8, device=device)
    actual = candidate.fused_add_mul_groupnorm(input1, input2, weight, bias, 4, eps=1e-5)
    expected = reference_fused_add_mul_groupnorm(input1, input2, weight, bias, 4, eps=1e-5)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)

    torch.manual_seed(4002)
    input1 = torch.randn(2, 6, 5, 3, device=device, dtype=torch.float64)
    input2 = torch.randn(1, 6, 1, 1, device=device, dtype=torch.float64)
    weight = torch.randn(6, device=device, dtype=torch.float64)
    bias = torch.randn(6, device=device, dtype=torch.float64)
    actual = candidate.fused_add_mul_groupnorm(input1, input2, weight, bias, 1, eps=1e-7)
    expected = reference_fused_add_mul_groupnorm(input1, input2, weight, bias, 1, eps=1e-7)
    assert torch.allclose(actual, expected, rtol=1e-8, atol=1e-8)

    torch.manual_seed(4003)
    input1 = torch.randn(2, 10, 3, 3, device=device)
    input2 = torch.randn(2, 10, 3, 3, device=device)
    out = torch.empty_like(input1)
    returned = candidate.fused_add_mul_groupnorm(input1, input2, None, None, 5, eps=1e-3, out=out)
    expected = reference_fused_add_mul_groupnorm(input1, input2, None, None, 5, eps=1e-3)
    assert returned is out
    assert torch.allclose(out, expected, rtol=2e-5, atol=2e-5)
