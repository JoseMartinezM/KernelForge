import torch

from pytorch_reference import mul_relu as reference_mul_relu


def unit_tests(candidate):
    assert torch.cuda.is_available()
    device = "cuda"

    torch.manual_seed(201)
    x = torch.randn(3, 5, 7, device=device, dtype=torch.float32)
    other = torch.randn(3, 5, 7, device=device, dtype=torch.float32)
    actual = candidate.mul_relu(x, other)
    expected = reference_mul_relu(x, other)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)

    torch.manual_seed(202)
    x16 = torch.randn(11, 3, device=device, dtype=torch.float16)
    other16 = torch.randn(11, 3, device=device, dtype=torch.float16)
    actual16 = candidate.mul_relu(x16, other16)
    expected16 = reference_mul_relu(x16, other16)
    assert torch.allclose(actual16, expected16, rtol=1e-3, atol=1e-3)

    torch.manual_seed(203)
    xb = torch.randn(2, 1, 9, device=device)
    ob = torch.randn(1, 4, 9, device=device)
    actual_b = candidate.mul_relu(xb, ob)
    expected_b = reference_mul_relu(xb, ob)
    assert torch.allclose(actual_b, expected_b, rtol=1e-6, atol=1e-6)

    x_inplace = torch.tensor([-2.0, 3.0, -0.5, 4.0], device=device)
    original = x_inplace.clone()
    other_inplace = torch.tensor([4.0, -2.0, -3.0, 0.25], device=device)
    actual_inplace = candidate.mul_relu(x_inplace, other_inplace, inplace=True)
    expected_inplace = reference_mul_relu(original, other_inplace, inplace=True)
    assert torch.allclose(actual_inplace, expected_inplace, rtol=1e-6, atol=1e-6)
    assert torch.allclose(x_inplace, original, rtol=0, atol=0)
