import torch

from pytorch_reference import log_softmax as reference_log_softmax


def _assert_forward_matches(
    candidate,
    x: torch.Tensor,
    *,
    dim=-1,
    dtype=None,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    expected = reference_log_softmax(x, dim=dim, dtype=dtype)
    actual = candidate.log_softmax(x, dim=dim, dtype=dtype)

    assert actual.shape == x.shape
    assert actual.dtype == expected.dtype
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def public_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"

    torch.manual_seed(1101)
    x = torch.randn(3, 7, device="cuda", dtype=torch.float32).contiguous()
    _assert_forward_matches(candidate, x)

    torch.manual_seed(1102)
    x = (torch.randn(2, 5, 3, device="cuda", dtype=torch.float32) * 2.5).contiguous()
    _assert_forward_matches(candidate, x, dim=1)

    torch.manual_seed(1103)
    x = torch.randn(4, 3, 9, device="cuda", dtype=torch.float16).contiguous()
    _assert_forward_matches(candidate, x, dim=-2, rtol=2e-2, atol=2e-2)
