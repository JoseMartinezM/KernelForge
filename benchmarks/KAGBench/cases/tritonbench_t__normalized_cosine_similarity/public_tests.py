import torch

from pytorch_reference import normalized_cosine_similarity as reference_normalized_cosine_similarity


def _assert_close(actual, expected, *, atol=1e-5, rtol=1e-5):
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"max diff={(actual - expected).abs().max().item()}\nactual={actual}\nexpected={expected}"
    )


def public_tests(candidate):
    assert torch.cuda.is_available()

    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    x2 = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device="cuda")
    expected = reference_normalized_cosine_similarity(x1, x2)
    actual = candidate.normalized_cosine_similarity(x1, x2)
    _assert_close(actual, expected)

    torch.manual_seed(0)
    x1 = torch.randn(2, 3, 4, device="cuda")
    x2 = torch.randn(2, 3, 4, device="cuda")
    expected = reference_normalized_cosine_similarity(x1, x2, dim=2)
    actual = candidate.normalized_cosine_similarity(x1, x2, dim=2)
    _assert_close(actual, expected)
