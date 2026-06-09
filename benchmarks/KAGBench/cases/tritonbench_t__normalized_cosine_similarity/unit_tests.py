import torch

from pytorch_reference import normalized_cosine_similarity as reference_normalized_cosine_similarity


def _assert_close(actual, expected, *, atol=1e-5, rtol=1e-5):
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"max diff={(actual - expected).abs().max().item()}\nactual={actual}\nexpected={expected}"
    )


def unit_tests(candidate):
    assert torch.cuda.is_available()

    torch.manual_seed(123)
    x1 = torch.randn(3, 5, 7, device="cuda")
    x2 = torch.randn(3, 5, 7, device="cuda")
    expected = reference_normalized_cosine_similarity(x1, x2)
    actual = candidate.normalized_cosine_similarity(x1, x2)
    _assert_close(actual, expected)

    torch.manual_seed(124)
    x1 = torch.randn(2, 3, 5, 7, device="cuda")
    x2 = torch.randn(2, 3, 5, 7, device="cuda")
    expected = reference_normalized_cosine_similarity(x1, x2, dim=3)
    actual = candidate.normalized_cosine_similarity(x1, x2, dim=3)
    _assert_close(actual, expected)

    x1 = torch.tensor(
        [[1.0, 0.0, -2.0], [0.5, -0.25, 0.75]], device="cuda"
    )
    x2 = torch.tensor(
        [[-1.0, 2.0, 0.0], [0.25, 0.5, -0.5]], device="cuda"
    )
    expected = reference_normalized_cosine_similarity(x1, x2, p_norm=1)
    actual = candidate.normalized_cosine_similarity(x1, x2, p_norm=1)
    _assert_close(actual, expected)

    x1 = torch.tensor([[1e-13, 0.0, 0.0], [0.0, 2e-13, 0.0]], device="cuda")
    x2 = torch.tensor([[0.0, 1e-13, 0.0], [0.0, 0.0, -3e-13]], device="cuda")
    expected = reference_normalized_cosine_similarity(
        x1, x2, eps_similarity=1e-6, eps_norm=1e-10
    )
    actual = candidate.normalized_cosine_similarity(
        x1, x2, eps_similarity=1e-6, eps_norm=1e-10
    )
    _assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    torch.manual_seed(125)
    x1 = torch.randn(4, 6, device="cuda", dtype=torch.float64)
    x2 = torch.randn(4, 6, device="cuda", dtype=torch.float64)
    expected = reference_normalized_cosine_similarity(x1, x2, dim=0)
    actual = candidate.normalized_cosine_similarity(x1, x2, dim=0)
    _assert_close(actual, expected, atol=1e-8, rtol=1e-8)
