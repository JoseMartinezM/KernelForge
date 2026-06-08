Implement the public function `_l2_norm_fwd(x, eps=1e-6)`.

The function receives a CUDA `torch.Tensor` whose last dimension is the feature dimension. It returns a tensor with the same shape and dtype as `x`, where each row (after flattening all leading dimensions) is divided by its L2 norm computed over the last dimension:

```python
y = x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)
```

For numerical behavior, compute the squared sum and reciprocal scale in float32 precision, then return values in the input dtype. The output shape must match the original input shape. Inputs may have one or more dimensions, odd feature sizes, all-zero rows, negative values, and non-contiguous layouts; the result should match the mathematical operation above for the logical tensor values.

The implementation should support the same API name and default epsilon as the source: `_l2_norm_fwd(x, eps=1e-6)`.
