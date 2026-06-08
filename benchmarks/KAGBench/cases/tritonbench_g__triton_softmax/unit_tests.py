import torch

from pytorch_reference import triton_softmax as reference_triton_softmax


def _assert_matches_reference(candidate, x: torch.Tensor, *, rtol: float, atol: float) -> None:
    expected = reference_triton_softmax(x)
    actual = candidate.triton_softmax(x)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def unit_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"

    torch.manual_seed(201)
    x = torch.randn(1, 16, device="cuda", dtype=torch.float32).contiguous()
    _assert_matches_reference(candidate, x, rtol=1e-4, atol=1e-4)

    torch.manual_seed(202)
    x = torch.randn(11, 1, device="cuda", dtype=torch.float32).contiguous()
    _assert_matches_reference(candidate, x, rtol=1e-4, atol=1e-4)

    torch.manual_seed(203)
    x = (torch.randn(7, 37, device="cuda", dtype=torch.float32) * 3.0 - 1.5).contiguous()
    _assert_matches_reference(candidate, x, rtol=1e-4, atol=1e-4)

    base = torch.linspace(-80.0, 80.0, steps=17, device="cuda", dtype=torch.float32)
    x = torch.stack(
        [
            base,
            -base,
            base + 1000.0,
            base - 1000.0,
            torch.sin(base) * 50.0,
        ],
        dim=0,
    ).contiguous()
    _assert_matches_reference(candidate, x, rtol=1e-4, atol=1e-4)

    torch.manual_seed(204)
    x = (torch.randn(3, 1024, device="cuda", dtype=torch.float32) * 2.0).contiguous()
    _assert_matches_reference(candidate, x, rtol=1e-4, atol=1e-4)

    torch.manual_seed(205)
    x = (torch.randn(9, 23, device="cuda", dtype=torch.float16) * 5.0).contiguous()
    _assert_matches_reference(candidate, x, rtol=2e-2, atol=2e-2)
