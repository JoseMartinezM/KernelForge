import torch

from pytorch_reference import masked_select as reference_masked_select


def _check(candidate, inp: torch.Tensor, mask: torch.Tensor) -> None:
    original_inp = inp.clone()
    original_mask = mask.clone()

    actual = candidate.masked_select(inp, mask)
    expected = reference_masked_select(inp, mask)

    assert actual.ndim == 1
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"dtype mismatch: {actual.dtype} != {expected.dtype}"
    assert actual.device == expected.device, f"device mismatch: {actual.device} != {expected.device}"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(inp, original_inp, rtol=0, atol=0)
    assert torch.equal(mask, original_mask), "candidate.masked_select must not modify mask"


def unit_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    # All-true mask with a non-power-of-two number of float32 elements.
    torch.manual_seed(201)
    inp = torch.randn(37, device="cuda", dtype=torch.float32)
    mask = torch.ones(37, device="cuda", dtype=torch.bool)
    _check(candidate, inp, mask)

    # All-false mask should produce an empty one-dimensional output.
    torch.manual_seed(202)
    inp = torch.randint(-50, 50, (2, 3, 5), device="cuda", dtype=torch.int64)
    mask = torch.zeros(2, 3, 5, device="cuda", dtype=torch.bool)
    _check(candidate, inp, mask)

    # Broadcasting a mask over a middle dimension.
    inp = torch.arange(2 * 3 * 4, device="cuda", dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor(
        [[[True, False, True, False]], [[False, True, False, True]]],
        device="cuda",
        dtype=torch.bool,
    )
    _check(candidate, inp, mask)

    # Higher-rank float tensor with an irregular selected count.
    torch.manual_seed(203)
    inp = torch.randn(2, 3, 4, 5, device="cuda", dtype=torch.float32)
    mask = torch.rand(2, 3, 4, 5, device="cuda") > 0.55
    _check(candidate, inp, mask)

    # Integer dtype with a non-power-of-two element count and mixed mask values.
    torch.manual_seed(204)
    inp = torch.randint(-1000, 1000, (7, 11), device="cuda", dtype=torch.int32)
    mask = torch.rand(7, 11, device="cuda") > 0.35
    _check(candidate, inp, mask)

    # Broadcast both operands to a common higher-rank shape.
    inp = torch.tensor([[-3.5, 0.0, 2.25, 8.0, -1.0]], device="cuda", dtype=torch.float32)
    mask = torch.tensor(
        [[[True], [False], [True]], [[False], [True], [True]]],
        device="cuda",
        dtype=torch.bool,
    )
    _check(candidate, inp, mask)
