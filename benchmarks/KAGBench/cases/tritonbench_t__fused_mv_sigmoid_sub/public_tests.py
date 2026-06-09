import torch
from pytorch_reference import fused_mv_sigmoid_sub as reference_fused_mv_sigmoid_sub


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required"
    torch.manual_seed(10)

    input = torch.randn(4, 3, device="cuda", dtype=torch.float32)
    vec = torch.randn(3, device="cuda", dtype=torch.float32)
    other = torch.randn(4, device="cuda", dtype=torch.float32)
    expected = reference_fused_mv_sigmoid_sub(input, vec, other)
    actual = candidate.fused_mv_sigmoid_sub(input, vec, other)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)

    input_scalar = torch.randn(5, 2, device="cuda", dtype=torch.float32)
    vec_scalar = torch.randn(2, device="cuda", dtype=torch.float32)
    expected_scalar = reference_fused_mv_sigmoid_sub(input_scalar, vec_scalar, 0.25, alpha=2)
    actual_scalar = candidate.fused_mv_sigmoid_sub(input_scalar, vec_scalar, 0.25, alpha=2)
    assert torch.allclose(actual_scalar, expected_scalar, rtol=1e-5, atol=1e-5)
