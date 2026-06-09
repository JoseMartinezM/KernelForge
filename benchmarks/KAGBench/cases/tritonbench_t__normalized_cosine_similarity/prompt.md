# Task

Implement `normalized_cosine_similarity(x1, x2, dim=1, eps_similarity=1e-08, p_norm=2, eps_norm=1e-12)`.

The function accepts two PyTorch tensors and returns their cosine similarity after first normalizing each input tensor independently.

Behavior:
1. Normalize `x1` with `torch.nn.functional.normalize(x1, p=p_norm, dim=dim, eps=eps_norm)`.
2. Normalize `x2` with `torch.nn.functional.normalize(x2, p=p_norm, dim=dim, eps=eps_norm)`.
3. Return `torch.nn.functional.cosine_similarity` of the two normalized tensors along `dim`, using `eps=eps_similarity`.

The public API must match the signature above, including default argument values. Inputs are CUDA tensors with broadcast-compatible shapes for the PyTorch operations.
