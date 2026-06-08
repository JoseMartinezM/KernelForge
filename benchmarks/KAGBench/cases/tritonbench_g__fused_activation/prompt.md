# Fused add, multiply, and sigmoid activation

Implement `fused_activation.py` with the following public API:

```python
def fused_add_mul_activation_torch(
    in_out_tensor: torch.Tensor,
    bias: torch.Tensor,
    in_tensor: torch.Tensor,
) -> torch.Tensor:
    ...
```

The inputs are contiguous one-dimensional CUDA floating-point tensors. `bias` is one-dimensional and non-empty, and `in_tensor` has the same number of elements as `in_out_tensor`.

Expected behavior:

- Update `in_out_tensor` in place and return the updated tensor.
- For each element index `i`, compute:

  ```text
  in_out_tensor[i] = sigmoid(original_in_out_tensor[i] + bias[i % bias.numel()] + 0.5 * in_tensor[i])
  ```

- `original_in_out_tensor[i]` means the value from `in_out_tensor` before any elements are updated.
- The returned tensor should alias the same storage as `in_out_tensor`.
- Do not modify `bias` or `in_tensor`.
- Preserve the shape, dtype, and CUDA device of `in_out_tensor`.
