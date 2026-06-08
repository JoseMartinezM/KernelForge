import torch

from pytorch_reference import wrapper as reference_wrapper


def _compare_wrapper(candidate, size_m: int, d_head: int, seed: int) -> None:
    torch.manual_seed(seed)
    expected = reference_wrapper(size_m, d_head)

    torch.manual_seed(seed)
    actual = candidate.wrapper(size_m, d_head)

    assert actual.shape == (d_head, size_m)
    assert actual.shape == expected.shape
    assert actual.dtype == torch.float16
    assert actual.dtype == expected.dtype
    assert actual.device.type == "cuda"
    assert torch.allclose(actual, expected, rtol=0.0, atol=0.0)


def test_square_baseline(candidate):
    _compare_wrapper(candidate, size_m=8, d_head=8, seed=101)


def test_rectangular_more_rows_than_columns(candidate):
    _compare_wrapper(candidate, size_m=65, d_head=17, seed=202)


def test_rectangular_more_columns_than_rows(candidate):
    _compare_wrapper(candidate, size_m=19, d_head=73, seed=303)


def test_odd_dimensions(candidate):
    _compare_wrapper(candidate, size_m=31, d_head=45, seed=404)


def test_dtype_is_fixed_float16_under_float32_default(candidate):
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        _compare_wrapper(candidate, size_m=23, d_head=11, seed=505)
    finally:
        torch.set_default_dtype(old_dtype)


def test_dtype_is_fixed_float16_under_float64_default(candidate):
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        _compare_wrapper(candidate, size_m=13, d_head=29, seed=606)
    finally:
        torch.set_default_dtype(old_dtype)


def test_repeated_deterministic_calls(candidate):
    size_m, d_head, seed = 37, 9, 707

    torch.manual_seed(seed)
    first = candidate.wrapper(size_m, d_head)

    torch.manual_seed(seed)
    second = candidate.wrapper(size_m, d_head)

    assert torch.equal(first, second)


def unit_tests(candidate):
    test_square_baseline(candidate)
    test_rectangular_more_rows_than_columns(candidate)
    test_rectangular_more_columns_than_rows(candidate)
    test_odd_dimensions(candidate)
    test_dtype_is_fixed_float16_under_float32_default(candidate)
    test_dtype_is_fixed_float16_under_float64_default(candidate)
    test_repeated_deterministic_calls(candidate)
