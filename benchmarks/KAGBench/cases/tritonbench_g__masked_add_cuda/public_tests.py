import torch

from pytorch_reference import masked_add as reference_masked_add


def _check(candidate, grad: torch.Tensor, p_data: torch.Tensor, p_mask: torch.Tensor, alpha: float) -> None:
    actual_grad = grad.clone()
    expected_grad = grad.clone()

    candidate.masked_add(actual_grad, p_data, p_mask, alpha=alpha)
    reference_masked_add(expected_grad, p_data, p_mask, alpha=alpha)

    assert actual_grad.shape == expected_grad.shape
    assert actual_grad.dtype == expected_grad.dtype
    assert torch.allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    torch.manual_seed(101)
    grad = torch.randn(128, device="cuda", dtype=torch.float32)
    p_data = torch.randn(128, device="cuda", dtype=torch.float32)
    p_mask = torch.randint(0, 2, (128,), device="cuda", dtype=torch.int32)
    _check(candidate, grad, p_data, p_mask, alpha=0.5)

    torch.manual_seed(102)
    grad = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    p_data = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    p_mask = torch.rand(4, 8, device="cuda") > 0.4
    _check(candidate, grad, p_data, p_mask, alpha=1.25)
