import torch

from pytorch_reference import fused_add_mul_activation_torch as reference_fused_activation


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.float16:
        return 3e-3, 3e-3
    return 1e-5, 1e-6


def _check_case(candidate, in_out_tensor: torch.Tensor, bias: torch.Tensor, in_tensor: torch.Tensor) -> None:
    actual_in_out = in_out_tensor.clone()
    expected_in_out = in_out_tensor.clone()
    original_bias = bias.clone()
    original_in_tensor = in_tensor.clone()

    actual_return = candidate.fused_add_mul_activation_torch(actual_in_out, bias, in_tensor)
    expected_return = reference_fused_activation(expected_in_out, bias, in_tensor)

    rtol, atol = _tolerances(in_out_tensor.dtype)
    assert actual_return.data_ptr() == actual_in_out.data_ptr(), "return value must alias in_out_tensor"
    assert expected_return.data_ptr() == expected_in_out.data_ptr()
    assert actual_in_out.shape == expected_in_out.shape
    assert actual_in_out.dtype == expected_in_out.dtype
    assert actual_in_out.device == expected_in_out.device
    torch.testing.assert_close(actual_in_out, expected_in_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_return, expected_return, rtol=rtol, atol=atol)
    torch.testing.assert_close(bias, original_bias, rtol=0, atol=0)
    torch.testing.assert_close(in_tensor, original_in_tensor, rtol=0, atol=0)


def unit_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    # Odd number of float32 elements with one bias value per element.
    torch.manual_seed(201)
    in_out_tensor = torch.randn(37, device="cuda", dtype=torch.float32).contiguous()
    bias = torch.randn(37, device="cuda", dtype=torch.float32).contiguous()
    in_tensor = torch.randn(37, device="cuda", dtype=torch.float32).contiguous()
    _check_case(candidate, in_out_tensor, bias, in_tensor)

    # Bias shorter than the data; the bias should repeat by element index modulo bias length.
    torch.manual_seed(202)
    in_out_tensor = torch.randn(257, device="cuda", dtype=torch.float32).contiguous()
    bias = torch.randn(16, device="cuda", dtype=torch.float32).contiguous()
    in_tensor = torch.randn(257, device="cuda", dtype=torch.float32).contiguous()
    _check_case(candidate, in_out_tensor, bias, in_tensor)

    # Explicit negative and positive values, including a short non-dividing bias length.
    torch.manual_seed(203)
    in_out_tensor = torch.tensor(
        [-6.0, -2.5, -0.25, 0.0, 0.75, 3.0, -1.5, 2.25, -4.0, 5.0],
        device="cuda",
        dtype=torch.float32,
    )
    bias = torch.tensor([-1.0, 0.5, 2.0], device="cuda", dtype=torch.float32)
    in_tensor = torch.tensor(
        [4.0, -3.0, 1.0, -2.0, 0.0, 6.0, -5.0, 2.0, 3.5, -4.0],
        device="cuda",
        dtype=torch.float32,
    )
    _check_case(candidate, in_out_tensor, bias, in_tensor)

    # Scalar-like bias length should still apply to every element.
    torch.manual_seed(204)
    in_out_tensor = torch.randn(96, device="cuda", dtype=torch.float32).contiguous() * 2.0 - 1.0
    bias = torch.tensor([0.375], device="cuda", dtype=torch.float32)
    in_tensor = torch.randn(96, device="cuda", dtype=torch.float32).contiguous()
    _check_case(candidate, in_out_tensor.contiguous(), bias, in_tensor)

    # Another non-power-of-two bias length to catch incorrect block-wise or quotient indexing.
    torch.manual_seed(205)
    in_out_tensor = torch.randn(511, device="cuda", dtype=torch.float32).contiguous()
    bias = torch.randn(5, device="cuda", dtype=torch.float32).contiguous()
    in_tensor = torch.randn(511, device="cuda", dtype=torch.float32).contiguous()
    _check_case(candidate, in_out_tensor, bias, in_tensor)

    # Float16 inputs use looser tolerances but should preserve dtype and in-place aliasing.
    torch.manual_seed(206)
    in_out_tensor = torch.randn(129, device="cuda", dtype=torch.float16).contiguous()
    bias = torch.randn(7, device="cuda", dtype=torch.float16).contiguous()
    in_tensor = torch.randn(129, device="cuda", dtype=torch.float16).contiguous()
    _check_case(candidate, in_out_tensor, bias, in_tensor)
