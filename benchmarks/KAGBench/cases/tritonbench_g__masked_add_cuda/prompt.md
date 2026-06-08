# Masked in-place add

Implement `masked_add(grad: torch.Tensor, p_data: torch.Tensor, p_mask: torch.Tensor, alpha: float = 0)`.

The function receives CUDA tensors `grad`, `p_data`, and `p_mask` with the same shape. It must update `grad` in place according to `p_mask`:

- Where `p_mask` is false or zero, set `grad` to `grad + alpha * p_data` for that element.
- Where `p_mask` is true or nonzero, leave the corresponding `grad` element unchanged.

The return value is not important; callers will inspect the mutated `grad` tensor after the function returns.

Expected behavior:

- Mutate `grad` in place rather than returning a separate result tensor.
- Treat boolean masks and integer masks consistently: false/zero means update, true/nonzero means keep the original `grad` value.
- Preserve the shape and dtype of `grad`.
- Support floating point CUDA tensors for `grad` and `p_data`.
