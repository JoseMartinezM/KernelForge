import torch

from pytorch_reference import sigmoid_adaptive_avg_pool2d as reference_sigmoid_adaptive_avg_pool2d


def _assert_close(actual, expected, *, atol=1e-5, rtol=1e-5):
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"max diff={(actual - expected).abs().max().item()}\nactual={actual}\nexpected={expected}"
    )


def unit_tests(candidate):
    assert torch.cuda.is_available()

    torch.manual_seed(210)
    input_tensor = torch.randn(3, 1, 7, 7, device="cuda")
    expected = reference_sigmoid_adaptive_avg_pool2d(input_tensor, 1)
    actual = candidate.sigmoid_adaptive_avg_pool2d(input_tensor, 1)
    _assert_close(actual, expected)

    torch.manual_seed(211)
    input_tensor = torch.randn(2, 4, 9, 9, device="cuda")
    expected = reference_sigmoid_adaptive_avg_pool2d(input_tensor, (2, 2))
    actual = candidate.sigmoid_adaptive_avg_pool2d(input_tensor, (2, 2))
    _assert_close(actual, expected)

    torch.manual_seed(212)
    input_tensor = torch.randn(1, 2, 8, 8, device="cuda") * 4.0 - 1.0
    expected = reference_sigmoid_adaptive_avg_pool2d(input_tensor, (4, 4))
    actual = candidate.sigmoid_adaptive_avg_pool2d(input_tensor, (4, 4))
    _assert_close(actual, expected)

    torch.manual_seed(213)
    input_tensor = torch.randn(2, 2, 8, 8, device="cuda", dtype=torch.float64)
    expected = reference_sigmoid_adaptive_avg_pool2d(input_tensor, (4, 4))
    actual = candidate.sigmoid_adaptive_avg_pool2d(input_tensor, (4, 4))
    _assert_close(actual, expected, atol=1e-8, rtol=1e-8)
