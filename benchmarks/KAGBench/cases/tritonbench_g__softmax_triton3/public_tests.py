import torch

from pytorch_reference import softmax as reference_softmax


def _check_softmax(
    candidate,
    input_tensor: torch.Tensor,
    mask: torch.Tensor = None,
    dim=-1,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    assert input_tensor.is_contiguous()
    original_input = input_tensor.clone()
    original_mask = mask.clone() if mask is not None else None

    actual = candidate.softmax(input_tensor, mask=mask, dim=dim)
    expected = reference_softmax(input_tensor, mask=mask, dim=dim)

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(input_tensor, original_input, rtol=0.0, atol=0.0)
    if mask is not None:
        assert mask.is_contiguous()
        torch.testing.assert_close(mask, original_mask, rtol=0.0, atol=0.0)


def public_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"

    torch.manual_seed(101)
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32).contiguous()
    _check_softmax(candidate, x)

    torch.manual_seed(102)
    x = (torch.randn(2, 3, 5, device="cuda", dtype=torch.float32) * 2.0).contiguous()
    mask = (torch.randn(2, 3, 5, device="cuda", dtype=torch.float32) * 0.75 - 0.25).contiguous()
    _check_softmax(candidate, x, mask=mask, dim=2)
