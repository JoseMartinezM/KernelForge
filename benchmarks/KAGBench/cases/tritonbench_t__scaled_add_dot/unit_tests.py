import torch
from pytorch_reference import scaled_add_dot as reference_scaled_add_dot


def _check_case(candidate, shape, alpha, dtype, seed, *, rtol, atol):
    torch.manual_seed(seed)
    y = torch.randn(shape, device="cuda", dtype=dtype)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    y_candidate = y.clone()
    y_reference = y.clone()
    out_candidate = candidate.scaled_add_dot(y_candidate, x, alpha)
    out_reference = reference_scaled_add_dot(y_reference, x, alpha)
    assert torch.allclose(y_candidate, y_reference, rtol=rtol, atol=atol)
    assert torch.allclose(out_candidate, out_reference, rtol=rtol, atol=atol)


def unit_tests(candidate):
    assert torch.cuda.is_available()
    _check_case(candidate, (1,), 0.0, torch.float32, 201, rtol=1e-6, atol=1e-6)
    _check_case(candidate, (39,), -2.25, torch.float32, 202, rtol=1e-5, atol=1e-5)
    _check_case(candidate, (1007,), 0.125, torch.float32, 203, rtol=1e-5, atol=1e-5)
    _check_case(candidate, (335,), -0.5, torch.float64, 204, rtol=1e-9, atol=1e-9)
