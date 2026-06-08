import torch

from pytorch_reference import fused_add_mul_activation_torch as reference_fused_activation


def _check_case(candidate, in_out_tensor: torch.Tensor, bias: torch.Tensor, in_tensor: torch.Tensor) -> None:
    actual_in_out = in_out_tensor.clone()
    expected_in_out = in_out_tensor.clone()
    original_bias = bias.clone()
    original_in_tensor = in_tensor.clone()

    actual_return = candidate.fused_add_mul_activation_torch(actual_in_out, bias, in_tensor)
    expected_return = reference_fused_activation(expected_in_out, bias, in_tensor)

    assert actual_return.data_ptr() == actual_in_out.data_ptr()
    assert expected_return.data_ptr() == expected_in_out.data_ptr()
    assert actual_in_out.shape == expected_in_out.shape
    assert actual_in_out.dtype == expected_in_out.dtype
    assert actual_in_out.device == expected_in_out.device
    torch.testing.assert_close(actual_in_out, expected_in_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_return, expected_return, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(bias, original_bias, rtol=0, atol=0)
    torch.testing.assert_close(in_tensor, original_in_tensor, rtol=0, atol=0)


def public_tests(candidate) -> None:
    assert torch.cuda.is_available(), "CUDA is required for these tests"

    torch.manual_seed(101)
    in_out_tensor = torch.randn(128, device="cuda", dtype=torch.float32).contiguous()
    bias = torch.randn(8, device="cuda", dtype=torch.float32).contiguous()
    in_tensor = torch.randn(128, device="cuda", dtype=torch.float32).contiguous()
    _check_case(candidate, in_out_tensor, bias, in_tensor)

    torch.manual_seed(102)
    in_out_tensor = torch.randn(1024, device="cuda", dtype=torch.float32).contiguous()
    bias = torch.randn(64, device="cuda", dtype=torch.float32).contiguous()
    in_tensor = torch.randn(1024, device="cuda", dtype=torch.float32).contiguous()
    _check_case(candidate, in_out_tensor, bias, in_tensor)
