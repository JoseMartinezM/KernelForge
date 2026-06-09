import torch

from pytorch_reference import log_softmax_linear as reference_log_softmax_linear


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"
    device = "cuda"

    input_a = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.25, 3.0]], device=device)
    weight_a = torch.tensor([[0.5, -1.0, 0.25], [1.5, 0.0, -0.5]], device=device)
    bias_a = torch.tensor([0.1, -0.2], device=device)
    expected_a = reference_log_softmax_linear(input_a, weight_a, bias_a)
    actual_a = candidate.log_softmax_linear(input_a, weight_a, bias_a)
    assert torch.allclose(actual_a, expected_a, rtol=1e-5, atol=1e-6)

    input_b = torch.tensor([[1.0, -2.0], [3.0, 0.5], [-1.5, 2.0]], device=device)
    weight_b = torch.tensor([[0.25, 0.75], [-0.5, 1.25], [1.0, -1.0]], device=device)
    expected_b = reference_log_softmax_linear(input_b, weight_b, dim=0)
    actual_b = candidate.log_softmax_linear(input_b, weight_b, dim=0)
    assert torch.allclose(actual_b, expected_b, rtol=1e-5, atol=1e-6)
