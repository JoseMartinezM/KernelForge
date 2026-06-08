import torch

from pytorch_reference import dropout as reference_dropout


def _check_dropout(candidate, x: torch.Tensor, x_keep: torch.Tensor, p: float) -> None:
    actual = candidate.dropout(x, x_keep, p)
    expected = reference_dropout(x, x_keep, p)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def public_tests(candidate) -> None:
    torch.manual_seed(1001)
    x = torch.randn(128, device="cuda", dtype=torch.float32).contiguous()
    x_keep = (torch.rand(128, device="cuda") > 0.35).to(torch.int32).contiguous()
    _check_dropout(candidate, x, x_keep, p=0.35)

    torch.manual_seed(1002)
    x = torch.randn(4, 16, device="cuda", dtype=torch.float32).contiguous()
    x_keep = (torch.rand(4, 16, device="cuda") > 0.2).to(torch.int32).contiguous()
    _check_dropout(candidate, x, x_keep, p=0.2)
