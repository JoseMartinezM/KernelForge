import torch

from pytorch_reference import add_gelu as reference_add_gelu


def unit_tests(candidate):
    assert torch.cuda.is_available()
    device = "cuda"

    torch.manual_seed(101)
    x = torch.randn(3, 5, 7, device=device, dtype=torch.float32)
    other = torch.randn(3, 5, 7, device=device, dtype=torch.float32)
    actual = candidate.add_gelu(x, other, alpha=-0.25, approximate="none")
    expected = reference_add_gelu(x, other, alpha=-0.25, approximate="none")
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)

    torch.manual_seed(102)
    x16 = torch.randn(9, 5, device=device, dtype=torch.float16)
    other16 = torch.randn(9, 5, device=device, dtype=torch.float16)
    actual16 = candidate.add_gelu(x16, other16, alpha=0.75, approximate="tanh")
    expected16 = reference_add_gelu(x16, other16, alpha=0.75, approximate="tanh")
    assert torch.allclose(actual16, expected16, rtol=2e-3, atol=2e-3)

    torch.manual_seed(103)
    xb = torch.randn(2, 1, 13, device=device)
    ob = torch.randn(1, 4, 13, device=device)
    actual_b = candidate.add_gelu(xb, ob, alpha=2.0)
    expected_b = reference_add_gelu(xb, ob, alpha=2.0)
    assert torch.allclose(actual_b, expected_b, rtol=1e-5, atol=1e-6)

    out = torch.empty_like(expected_b)
    returned = candidate.add_gelu(xb, ob, alpha=2.0, out=out)
    assert returned is out
    assert torch.allclose(out, expected_b, rtol=1e-5, atol=1e-6)
