import torch

from pytorch_reference import triton_softmax as reference_triton_softmax


def _assert_matches_reference(candidate, x: torch.Tensor, *, rtol: float = 1e-4, atol: float = 1e-4) -> None:
    expected = reference_triton_softmax(x)
    actual = candidate.triton_softmax(x)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def public_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"

    torch.manual_seed(101)
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32).contiguous()
    _assert_matches_reference(candidate, x)

    torch.manual_seed(102)
    x = torch.randn(12, 64, device="cuda", dtype=torch.float32).contiguous()
    _assert_matches_reference(candidate, x)
