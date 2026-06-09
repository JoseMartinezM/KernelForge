import torch
from pytorch_reference import scaled_add_dot as reference_scaled_add_dot


def _check_case(candidate, y, x, alpha, *, rtol=1e-5, atol=1e-5):
    y_candidate = y.clone()
    y_reference = y.clone()
    out_candidate = candidate.scaled_add_dot(y_candidate, x, alpha)
    out_reference = reference_scaled_add_dot(y_reference, x, alpha)
    assert torch.allclose(y_candidate, y_reference, rtol=rtol, atol=atol)
    assert torch.allclose(out_candidate, out_reference, rtol=rtol, atol=atol)


def public_tests(candidate):
    assert torch.cuda.is_available()
    device = "cuda"
    _check_case(
        candidate,
        torch.tensor([1.0, 2.0, 3.0, 4.0], device=device),
        torch.tensor([0.5, -1.0, 2.0, 0.25], device=device),
        1.5,
    )
    _check_case(
        candidate,
        torch.linspace(-3.0, 3.0, 129, device=device),
        torch.randn(129, device=device),
        -0.25,
    )
