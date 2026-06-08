# Mean reduction

Implement `mean_dim(x, dim, keepdim=False, *, dtype=None)`.

The function receives a CUDA `torch.Tensor` `x` and returns the arithmetic mean of `x` over one or more dimensions.

API requirements:

- `dim` may be an integer dimension or a list/tuple of dimensions.
- Negative dimensions should be interpreted in the standard PyTorch way.
- If `keepdim` is `False`, the reduced dimensions are removed from the result shape.
- If `keepdim` is `True`, the reduced dimensions remain with size 1.
- If `dtype` is provided, the returned tensor should have that dtype and the mean should match `torch.mean(x, dim=dim, keepdim=keepdim, dtype=dtype)`.
- If `dtype` is omitted, use the input tensor dtype, matching PyTorch mean semantics for floating-point inputs.

Your implementation should expose the public function `mean_dim` with the signature above.
