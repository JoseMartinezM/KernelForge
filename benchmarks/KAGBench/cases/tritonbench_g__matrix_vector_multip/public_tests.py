import torch

from pytorch_reference import mv as reference_mv


def _assert_mv_close(candidate, inp: torch.Tensor, vec: torch.Tensor) -> None:
    actual = candidate.mv(inp, vec)
    expected = reference_mv(inp, vec)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def public_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for matrix-vector tests"

    torch.manual_seed(101)
    inp = torch.randn(4, 3, device="cuda", dtype=torch.float32)
    vec = torch.randn(3, device="cuda", dtype=torch.float32)
    _assert_mv_close(candidate, inp, vec)

    torch.manual_seed(102)
    inp = torch.randn(32, 16, device="cuda", dtype=torch.float32)
    vec = torch.randn(16, device="cuda", dtype=torch.float32)
    _assert_mv_close(candidate, inp, vec)
