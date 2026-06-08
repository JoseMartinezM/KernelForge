import torch

from pytorch_reference import mv as reference_mv


def _assert_mv_close(
    candidate,
    inp: torch.Tensor,
    vec: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    actual = candidate.mv(inp, vec)
    expected = reference_mv(inp, vec)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def unit_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for matrix-vector tests"

    torch.manual_seed(201)
    inp = torch.randn(17, 13, device="cuda", dtype=torch.float32)
    vec = torch.randn(13, device="cuda", dtype=torch.float32)
    _assert_mv_close(candidate, inp, vec, rtol=1e-4, atol=1e-4)

    torch.manual_seed(202)
    inp = -torch.rand(9, 27, device="cuda", dtype=torch.float32) * 3.0
    vec = torch.linspace(-2.0, 2.0, steps=27, device="cuda", dtype=torch.float32)
    _assert_mv_close(candidate, inp, vec, rtol=1e-4, atol=1e-4)

    torch.manual_seed(203)
    inp = torch.randn(15, 11, device="cuda", dtype=torch.float16)
    vec = torch.randn(11, device="cuda", dtype=torch.float16)
    _assert_mv_close(candidate, inp, vec, rtol=1e-2, atol=1e-2)

    torch.manual_seed(204)
    base_inp = torch.randn(7, 19, device="cuda", dtype=torch.float32)
    inp = base_inp.t()
    base_vec = torch.randn(14, device="cuda", dtype=torch.float32)
    vec = base_vec[::2]
    assert inp.shape == (19, 7)
    assert vec.shape == (7,)
    assert not inp.is_contiguous()
    assert not vec.is_contiguous()
    _assert_mv_close(candidate, inp, vec, rtol=1e-4, atol=1e-4)
