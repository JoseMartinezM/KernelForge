import torch

from pytorch_reference import softplus_linear as reference_softplus_linear


def public_tests(candidate):
    assert torch.cuda.is_available(), "CUDA is required for these tests"
    device = "cuda"

    input_a = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.25, 3.0]], device=device)
    weight_a = torch.tensor([[0.5, -1.0, 0.25], [1.5, 0.0, -0.5]], device=device)
    bias_a = torch.tensor([0.1, -0.2], device=device)
    expected_a = reference_softplus_linear(input_a, weight_a, bias_a)
    actual_a = candidate.softplus_linear(input_a, weight_a, bias_a)
    assert torch.allclose(actual_a, expected_a, rtol=1e-5, atol=1e-6)

    input_b = torch.tensor([[-3.0, 1.0], [0.25, 2.0], [4.0, -1.5]], device=device)
    weight_b = torch.tensor([[1.25, -0.75], [-0.5, 0.5], [0.25, 1.5]], device=device)
    expected_b = reference_softplus_linear(input_b, weight_b, beta=2.0, threshold=8.0)
    actual_b = candidate.softplus_linear(input_b, weight_b, beta=2.0, threshold=8.0)
    assert torch.allclose(actual_b, expected_b, rtol=1e-5, atol=1e-6)
