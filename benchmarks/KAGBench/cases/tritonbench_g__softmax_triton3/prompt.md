# Additive-mask softmax

Implement `softmax_triton3.py` with the following public API:

```python
def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    ...
```

`input` is a contiguous CUDA floating-point tensor. The function returns a tensor with the same shape, dtype, and device as `input`.

Compute softmax over the last dimension only. The `dim` argument is guaranteed to refer to the last dimension, either as `-1` or as the positive index `input.ndim - 1`.

If `mask` is provided, it is a contiguous CUDA floating-point tensor with the same shape as `input`. Treat it as an additive mask: add it elementwise to `input` before computing the softmax. It is not a boolean keep/drop mask.

Expected behavior:

- With no mask, match `torch.softmax(input, dim=-1)`.
- With a mask, match `torch.softmax(input + mask, dim=-1)`.
- Be numerically stable for large positive and negative input values.
- Do not modify `input` or `mask` in place.
