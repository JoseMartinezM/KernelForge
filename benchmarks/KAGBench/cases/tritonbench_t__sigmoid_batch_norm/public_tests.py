import torch

from pytorch_reference import sigmoid_batch_norm as reference_sigmoid_batch_norm


def public_tests(candidate):
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    device = "cuda"

    x = torch.randn(8, 5, device=device)
    running_mean = torch.randn(5, device=device) * 0.2
    running_var = torch.rand(5, device=device) + 0.5
    weight = torch.randn(5, device=device) * 0.3 + 1.0
    bias = torch.randn(5, device=device) * 0.1
    actual = candidate.sigmoid_batch_norm(x, running_mean, running_var, weight, bias)
    expected = reference_sigmoid_batch_norm(x, running_mean, running_var, weight, bias)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)

    x3 = torch.randn(3, 4, 7, device=device)
    running_mean3 = torch.linspace(-0.2, 0.3, 4, device=device)
    running_var3 = torch.linspace(0.7, 1.4, 4, device=device)
    actual3 = candidate.sigmoid_batch_norm(x3, running_mean3, running_var3, eps=1e-3)
    expected3 = reference_sigmoid_batch_norm(x3, running_mean3, running_var3, eps=1e-3)
    assert torch.allclose(actual3, expected3, rtol=1e-5, atol=1e-6)
