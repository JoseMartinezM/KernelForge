import torch

from pytorch_reference import triton_matmul as reference_triton_matmul


def _assert_matmul_close(
    candidate,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    a_original = a.clone()
    b_original = b.clone()
    expected = reference_triton_matmul(a, b)
    actual = candidate.triton_matmul(a, b)

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == (a.shape[0], b.shape[1])
    assert actual.dtype == a.dtype
    assert actual.device == a.device
    torch.testing.assert_close(a, a_original, rtol=0, atol=0)
    torch.testing.assert_close(b, b_original, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def public_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for matrix multiplication tests"

    torch.manual_seed(3101)
    a = torch.randn((16, 16), device="cuda", dtype=torch.float32)
    b = torch.randn((16, 16), device="cuda", dtype=torch.float32)
    _assert_matmul_close(candidate, a, b)

    torch.manual_seed(3102)
    a = torch.randn((64, 64), device="cuda", dtype=torch.float32)
    b = torch.randn((64, 64), device="cuda", dtype=torch.float32)
    _assert_matmul_close(candidate, a, b)
