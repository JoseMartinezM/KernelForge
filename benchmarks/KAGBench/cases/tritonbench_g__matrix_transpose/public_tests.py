import torch

from pytorch_reference import wrapper as reference_wrapper


def _run_case(candidate, size_m: int, d_head: int, seed: int) -> None:
    torch.manual_seed(seed)
    expected = reference_wrapper(size_m, d_head)

    torch.manual_seed(seed)
    actual = candidate.wrapper(size_m, d_head)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.is_cuda
    assert torch.allclose(actual, expected, rtol=0.0, atol=0.0)


def public_tests(candidate):
    _run_case(candidate, size_m=16, d_head=16, seed=0)
    _run_case(candidate, size_m=32, d_head=64, seed=1)
