import torch

from pytorch_reference import masked_select as reference_masked_select


def _check(candidate, inp: torch.Tensor, mask: torch.Tensor) -> None:
    actual = candidate.masked_select(inp, mask)
    expected = reference_masked_select(inp, mask)

    assert actual.ndim == 1
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    torch.manual_seed(101)
    inp = torch.randn(4, 5, device="cuda", dtype=torch.float32)
    mask = torch.rand(4, 5, device="cuda") > 0.4
    _check(candidate, inp, mask)

    torch.manual_seed(102)
    inp = torch.randint(-20, 20, (3, 4, 5), device="cuda", dtype=torch.int64)
    mask = torch.rand(3, 4, 5, device="cuda") > 0.6
    _check(candidate, inp, mask)
