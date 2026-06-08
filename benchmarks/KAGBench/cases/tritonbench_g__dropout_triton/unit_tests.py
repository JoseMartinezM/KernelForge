import torch

from pytorch_reference import dropout as reference_dropout


def _assert_dropout_close(
    candidate,
    x: torch.Tensor,
    x_keep: torch.Tensor,
    p: float,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    actual = candidate.dropout(x, x_keep, p)
    expected = reference_dropout(x, x_keep, p)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.is_cuda
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def unit_tests(candidate) -> None:
    torch.manual_seed(2001)
    x = torch.randn(1031, device="cuda", dtype=torch.float32).contiguous()
    x_keep = (torch.arange(1031, device="cuda") % 3 != 0).to(torch.int32).contiguous()
    _assert_dropout_close(candidate, x, x_keep, p=0.4)

    torch.manual_seed(2002)
    x = torch.randn(257, device="cuda", dtype=torch.float32).contiguous()
    x_keep = torch.ones_like(x, dtype=torch.int32).contiguous()
    _assert_dropout_close(candidate, x, x_keep, p=0.0)

    torch.manual_seed(2003)
    x = torch.randn(65, device="cuda", dtype=torch.float32).contiguous()
    all_drop = torch.zeros(65, device="cuda", dtype=torch.int32).contiguous()
    alternating = (torch.arange(65, device="cuda") % 2 == 0).to(torch.int32).contiguous()
    _assert_dropout_close(candidate, x, all_drop, p=0.25)
    _assert_dropout_close(candidate, x, alternating, p=0.25)

    torch.manual_seed(2004)
    x = torch.randn(2, 3, 5, 7, device="cuda", dtype=torch.float32).contiguous()
    x_keep = (torch.rand(2, 3, 5, 7, device="cuda") > 0.6).to(torch.bool).contiguous()
    _assert_dropout_close(candidate, x, x_keep, p=0.6)

    torch.manual_seed(2005)
    x = torch.randn(513, device="cuda", dtype=torch.float16).contiguous()
    x_keep = (torch.arange(513, device="cuda") % 5 < 3).to(torch.int32).contiguous()
    _assert_dropout_close(candidate, x, x_keep, p=0.3, rtol=1e-3, atol=1e-3)
