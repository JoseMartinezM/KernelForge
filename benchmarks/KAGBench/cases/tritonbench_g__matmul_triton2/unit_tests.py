import torch

from pytorch_reference import triton_matmul as reference_triton_matmul


def _assert_matmul_close(
    candidate,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    rtol: float,
    atol: float,
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


def unit_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for matrix multiplication tests"

    torch.manual_seed(4201)
    a = torch.randn((7, 13), device="cuda", dtype=torch.float32)
    b = torch.randn((13, 5), device="cuda", dtype=torch.float32)
    _assert_matmul_close(candidate, a, b, rtol=2e-2, atol=2e-2)

    torch.manual_seed(4202)
    a = torch.randn((33, 65), device="cuda", dtype=torch.float32)
    b = torch.randn((65, 17), device="cuda", dtype=torch.float32)
    _assert_matmul_close(candidate, a, b, rtol=2e-2, atol=2e-2)

    torch.manual_seed(4203)
    a = torch.rand((19, 23), device="cuda", dtype=torch.float32) * 4.0 - 2.0
    b = torch.rand((23, 29), device="cuda", dtype=torch.float32) * 6.0 - 3.0
    _assert_matmul_close(candidate, a.contiguous(), b.contiguous(), rtol=2e-2, atol=2e-2)

    torch.manual_seed(4204)
    a = torch.randn((15, 21), device="cuda", dtype=torch.float16)
    b = torch.randn((21, 10), device="cuda", dtype=torch.float16)
    _assert_matmul_close(candidate, a, b, rtol=1e-2, atol=1e-2)
