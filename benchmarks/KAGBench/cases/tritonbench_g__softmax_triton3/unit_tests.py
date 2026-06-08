import torch

from pytorch_reference import softmax as reference_softmax


def _check_softmax(
    candidate,
    input_tensor: torch.Tensor,
    mask: torch.Tensor = None,
    dim=-1,
    *,
    rtol: float,
    atol: float,
) -> None:
    assert input_tensor.is_contiguous()
    if mask is not None:
        assert mask.shape == input_tensor.shape
        assert mask.is_contiguous()

    original_input = input_tensor.clone()
    original_mask = mask.clone() if mask is not None else None

    actual = candidate.softmax(input_tensor, mask=mask, dim=dim)
    expected = reference_softmax(input_tensor, mask=mask, dim=dim)

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"dtype mismatch: {actual.dtype} != {expected.dtype}"
    assert actual.device == expected.device, f"device mismatch: {actual.device} != {expected.device}"
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(input_tensor, original_input, rtol=0.0, atol=0.0)
    if mask is not None:
        torch.testing.assert_close(mask, original_mask, rtol=0.0, atol=0.0)


def unit_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"

    # Odd, non-power-of-two last dimension without a mask.
    torch.manual_seed(201)
    x = (torch.randn(7, 37, device="cuda", dtype=torch.float32) * 3.0).contiguous()
    _check_softmax(candidate, x, rtol=1e-4, atol=1e-4)

    # Higher-rank input with a structured additive mask and positive last-dimension index.
    torch.manual_seed(202)
    x = torch.randn(2, 3, 4, 17, device="cuda", dtype=torch.float32).contiguous()
    mask = torch.zeros_like(x)
    mask[..., ::3] = -4.0
    mask[..., 1::3] = 1.25
    mask = mask.contiguous()
    _check_softmax(candidate, x, mask=mask, dim=x.ndim - 1, rtol=1e-4, atol=1e-4)

    # Large values should remain numerically stable.
    base = torch.linspace(-80.0, 80.0, steps=19, device="cuda", dtype=torch.float32)
    stable = torch.stack(
        [
            base,
            -base,
            base + 1000.0,
            base - 1000.0,
            torch.sin(base) * 60.0,
        ],
        dim=0,
    ).contiguous()
    _check_softmax(candidate, stable, rtol=1e-4, atol=1e-4)

    # Additive mask values are part of the logits, including large negative offsets.
    logits = torch.tensor(
        [
            [1000.0, 1001.0, 999.0, 995.0, 1002.0, 998.0, 996.0],
            [-1000.0, -999.5, -1001.0, -998.0, -1002.0, -997.5, -1003.0],
            [-20.0, -5.0, 0.0, 5.0, 20.0, 25.0, 30.0],
        ],
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    additive_mask = torch.tensor(
        [
            [0.0, -2.0, 1.0, -5.0, 0.5, -3.0, 2.0],
            [-1.5, 0.0, -4.0, 2.5, -6.0, 1.0, -8.0],
            [-20.0, -10.0, 0.0, 1.0, -1.0, -15.0, -30.0],
        ],
        device="cuda",
        dtype=torch.float32,
    ).contiguous()
    _check_softmax(candidate, logits, mask=additive_mask, rtol=1e-4, atol=1e-4)

    # Float16 higher-rank shape with an odd last dimension uses lower-precision tolerances.
    torch.manual_seed(203)
    x = (torch.randn(3, 5, 23, device="cuda", dtype=torch.float16) * 4.0).contiguous()
    mask = (torch.randn(3, 5, 23, device="cuda", dtype=torch.float16) * 0.5 - 0.25).contiguous()
    _check_softmax(candidate, x, mask=mask, rtol=2e-2, atol=2e-2)
