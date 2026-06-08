import torch

from pytorch_reference import masked_add as reference_masked_add


def _check(candidate, grad: torch.Tensor, p_data: torch.Tensor, p_mask: torch.Tensor, alpha: float) -> None:
    actual_grad = grad.clone()
    expected_grad = grad.clone()
    original_p_data = p_data.clone()
    original_p_mask = p_mask.clone()

    candidate.masked_add(actual_grad, p_data, p_mask, alpha=alpha)
    reference_masked_add(expected_grad, p_data, p_mask, alpha=alpha)

    assert actual_grad.shape == expected_grad.shape
    assert actual_grad.dtype == expected_grad.dtype
    assert torch.allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(p_data, original_p_data, rtol=0, atol=0)
    assert torch.equal(p_mask, original_p_mask)


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    # Odd number of elements with an integer mask.
    torch.manual_seed(201)
    grad = torch.randn(37, device="cuda", dtype=torch.float32)
    p_data = torch.randn(37, device="cuda", dtype=torch.float32)
    p_mask = torch.randint(0, 2, (37,), device="cuda", dtype=torch.int32)
    _check(candidate, grad, p_data, p_mask, alpha=0.75)

    # alpha=0 should leave grad unchanged regardless of the mask.
    torch.manual_seed(202)
    grad = torch.randn(65, device="cuda", dtype=torch.float32)
    p_data = torch.randn(65, device="cuda", dtype=torch.float32)
    p_mask = torch.randint(0, 2, (65,), device="cuda", dtype=torch.int64)
    _check(candidate, grad, p_data, p_mask, alpha=0.0)

    # Negative alpha with a boolean mask.
    torch.manual_seed(203)
    grad = torch.randn(53, device="cuda", dtype=torch.float32)
    p_data = torch.randn(53, device="cuda", dtype=torch.float32)
    p_mask = torch.rand(53, device="cuda") > 0.35
    _check(candidate, grad, p_data, p_mask, alpha=-1.5)

    # All-zero mask updates every element.
    torch.manual_seed(204)
    grad = torch.randn(40, device="cuda", dtype=torch.float32)
    p_data = torch.randn(40, device="cuda", dtype=torch.float32)
    p_mask = torch.zeros(40, device="cuda", dtype=torch.int32)
    _check(candidate, grad, p_data, p_mask, alpha=0.25)

    # All-one mask leaves every element unchanged.
    torch.manual_seed(205)
    grad = torch.randn(41, device="cuda", dtype=torch.float32)
    p_data = torch.randn(41, device="cuda", dtype=torch.float32)
    p_mask = torch.ones(41, device="cuda", dtype=torch.int32)
    _check(candidate, grad, p_data, p_mask, alpha=2.0)

    # Integer masks treat any nonzero value as masked out.
    grad = torch.linspace(-3.0, 4.0, 8, device="cuda", dtype=torch.float32)
    p_data = torch.linspace(0.5, 4.0, 8, device="cuda", dtype=torch.float32)
    p_mask = torch.tensor([0, 1, 2, -3, 0, 5, 0, -1], device="cuda", dtype=torch.int32)
    _check(candidate, grad, p_data, p_mask, alpha=1.1)

    # Higher-rank contiguous tensor.
    torch.manual_seed(206)
    grad = torch.randn(2, 3, 5, 7, device="cuda", dtype=torch.float32)
    p_data = torch.randn(2, 3, 5, 7, device="cuda", dtype=torch.float32)
    p_mask = torch.randint(0, 2, (2, 3, 5, 7), device="cuda", dtype=torch.int32)
    assert grad.is_contiguous() and p_data.is_contiguous() and p_mask.is_contiguous()
    _check(candidate, grad, p_data, p_mask, alpha=0.6)
