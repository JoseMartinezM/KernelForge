import torch

from pytorch_reference import softmax as reference_softmax


def _assert_matches_reference(candidate, x: torch.Tensor, *, rtol: float = 1e-4, atol: float = 1e-4) -> None:
    expected = reference_softmax(x)
    actual = candidate.softmax(x)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.device == x.device
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def public_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"

    torch.manual_seed(101)
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32).contiguous()
    _assert_matches_reference(candidate, x)

    torch.manual_seed(102)
    x = (torch.randn(16, 64, device="cuda", dtype=torch.float32) * 2.0).contiguous()
    _assert_matches_reference(candidate, x)
