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


def _assert_backward_matches(
    candidate,
    x: torch.Tensor,
    *,
    dim=-1,
    dtype=None,
    grad_output: torch.Tensor | None = None,
    forward_rtol: float = 1e-4,
    forward_atol: float = 1e-4,
    grad_rtol: float = 1e-4,
    grad_atol: float = 1e-4,
) -> None:
    candidate_x = x.detach().clone().requires_grad_(True)
    reference_x = x.detach().clone().requires_grad_(True)

    actual = candidate.log_softmax(candidate_x, dim=dim, dtype=dtype)
    expected = reference_log_softmax(reference_x, dim=dim, dtype=dtype)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=forward_rtol, atol=forward_atol)

    if grad_output is None:
        actual.sum().backward()
        expected.sum().backward()
    else:
        grad = grad_output.to(device=x.device, dtype=actual.dtype).contiguous()
        actual.backward(grad)
        expected.backward(grad)

    assert candidate_x.grad is not None
    assert reference_x.grad is not None
    torch.testing.assert_close(candidate_x.grad, reference_x.grad, rtol=grad_rtol, atol=grad_atol)


def unit_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"

    # Odd shape with the default last-dimension reduction.
    torch.manual_seed(2101)
    x = (torch.randn(3, 5, 7, device="cuda", dtype=torch.float32) * 3.0).contiguous()
    _assert_forward_matches(candidate, x)

    # Reduction over a non-last dimension.
    torch.manual_seed(2102)
    x = torch.randn(2, 7, 3, 5, device="cuda", dtype=torch.float32).contiguous()
    _assert_forward_matches(candidate, x, dim=1)

    # Negative dimension that is not the last dimension.
    torch.manual_seed(2103)
    x = (torch.randn(4, 3, 11, device="cuda", dtype=torch.float32) - 0.5).contiguous()
    _assert_forward_matches(candidate, x, dim=-2)

    # Float16 input with explicit float32 output dtype.
    torch.manual_seed(2104)
    x = (torch.randn(3, 5, 9, device="cuda", dtype=torch.float16) * 4.0).contiguous()
    _assert_forward_matches(candidate, x, dim=-1, dtype=torch.float32, rtol=1e-4, atol=1e-4)

    # Numerically stable behavior for very large values.
    large = torch.tensor(
        [
            [10000.0, 10001.0, 9999.0, 10003.0, 9998.0],
            [-10000.0, -9998.0, -10002.0, -9999.0, -10001.0],
            [5000.0, -5000.0, 0.0, 4999.0, -4999.0],
        ],
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    _assert_forward_matches(candidate, large, dim=-1)

    # Backward through a summed scalar loss.
    torch.manual_seed(2105)
    x = torch.randn(5, 4, 6, device="cuda", dtype=torch.float32).contiguous()
    _assert_backward_matches(candidate, x, dim=-1)

    # Backward with a fixed upstream gradient and non-last reduction dimension.
    torch.manual_seed(2106)
    x = torch.randn(2, 5, 3, device="cuda", dtype=torch.float32).contiguous()
    grad_output = torch.linspace(-0.75, 0.85, steps=x.numel(), device="cuda", dtype=torch.float32).reshape_as(x)
    _assert_backward_matches(candidate, x, dim=1, grad_output=grad_output)
