import torch

from pytorch_reference import kldivergence as reference_kldivergence


def _check_case(
    candidate,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    actual = candidate.kldivergence(x, y)
    expected = reference_kldivergence(x, y)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def unit_tests(candidate) -> None:
    torch.manual_seed(9101)
    x = torch.rand(3337, device="cuda", dtype=torch.float32) + 0.2
    y = torch.rand(3337, device="cuda", dtype=torch.float32) + 0.2
    _check_case(candidate, x, y, rtol=1e-4, atol=1e-5)

    torch.manual_seed(1121)
    x = torch.rand((2, 3, 5, 7), device="cuda", dtype=torch.float32).contiguous() + 0.1
    y = torch.rand((2, 3, 5, 7), device="cuda", dtype=torch.float32).contiguous() + 0.1
    _check_case(candidate, x, y, rtol=1e-4, atol=1e-5)

    torch.manual_seed(3141)
    x = torch.linspace(1.0e-7, 1.0e-3, 4099, device="cuda", dtype=torch.float32)
    y = torch.linspace(2.0e-7, 2.0e-3, 4099, device="cuda", dtype=torch.float32).flip(0)
    _check_case(candidate, x, y, rtol=1e-4, atol=1e-7)

    torch.manual_seed(5161)
    x = torch.rand(2051, device="cuda", dtype=torch.float16) * 2.0 + 0.25
    y = torch.rand(2051, device="cuda", dtype=torch.float16) * 2.0 + 0.25
    _check_case(candidate, x, y, rtol=2e-2, atol=2e-2)
