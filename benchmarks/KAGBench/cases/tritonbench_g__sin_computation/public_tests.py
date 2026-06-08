import torch

from pytorch_reference import sin_triton as reference_sin_triton


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)


def _check(candidate, x: torch.Tensor) -> None:
    original_x = x.clone()
    actual_out = torch.empty_like(x)
    expected_out = torch.empty_like(x)

    reference_sin_triton(original_x, expected_out)
    candidate.sin_triton(x, actual_out)

    _assert_close(actual_out, expected_out)
    assert torch.equal(x, original_x), "candidate.sin_triton must not modify x"


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    x = torch.tensor([0.0, 1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
    _check(candidate, x)

    torch.manual_seed(101)
    matrix = torch.rand((32, 32), device="cuda", dtype=torch.float32) * 6.0
    _check(candidate, matrix)
