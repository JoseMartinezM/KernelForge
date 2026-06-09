import torch
from pytorch_reference import fused_mv_sigmoid_sub as reference_fused_mv_sigmoid_sub


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required"

    torch.manual_seed(2024)
    input_odd = torch.randn(7, 11, device="cuda", dtype=torch.float32)
    vec_odd = torch.randn(11, device="cuda", dtype=torch.float32)
    other_odd = torch.randn(7, device="cuda", dtype=torch.float32)
    expected_odd = reference_fused_mv_sigmoid_sub(input_odd, vec_odd, other_odd, alpha=0.5)
    actual_odd = candidate.fused_mv_sigmoid_sub(input_odd, vec_odd, other_odd, alpha=0.5)
    assert torch.allclose(actual_odd, expected_odd, rtol=1e-5, atol=1e-5)

    torch.manual_seed(2025)
    input_scalar = torch.randn(3, 8, device="cuda", dtype=torch.float64)
    vec_scalar = torch.randn(8, device="cuda", dtype=torch.float64)
    expected_scalar = reference_fused_mv_sigmoid_sub(input_scalar, vec_scalar, -0.125, alpha=3)
    actual_scalar = candidate.fused_mv_sigmoid_sub(input_scalar, vec_scalar, -0.125, alpha=3)
    assert torch.allclose(actual_scalar, expected_scalar, rtol=1e-10, atol=1e-10)

    torch.manual_seed(2026)
    input_out = torch.randn(9, 5, device="cuda", dtype=torch.float32)
    vec_out = torch.randn(5, device="cuda", dtype=torch.float32)
    other_out = torch.linspace(-0.3, 0.3, 9, device="cuda", dtype=torch.float32)
    out = torch.empty(9, device="cuda", dtype=torch.float32)
    expected_out = reference_fused_mv_sigmoid_sub(input_out, vec_out, other_out, alpha=1.25)
    actual_out = candidate.fused_mv_sigmoid_sub(input_out, vec_out, other_out, alpha=1.25, out=out)
    assert actual_out is out
    assert torch.allclose(out, expected_out, rtol=1e-5, atol=1e-5)
