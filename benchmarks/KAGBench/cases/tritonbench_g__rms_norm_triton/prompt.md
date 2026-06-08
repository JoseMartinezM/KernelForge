Implement the public function `rms_norm(x: torch.Tensor, normalized_shape, weight: torch.Tensor, eps=1e-5) -> torch.Tensor`.

The function receives a CUDA `torch.Tensor` `x`, a `normalized_shape`, and a one-dimensional `weight` tensor. For the tested cases, `normalized_shape` is a one-element tuple equal to the size of the last dimension of `x`, and `weight` has that same length.

Return a tensor with the same shape and dtype as `x`. Normalize each row over the last dimension using RMS normalization, then apply the per-feature weight:

```python
scale = torch.rsqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + eps)
y = x * scale * weight
```

For numerical behavior, compute the squared mean and reciprocal scale in float32 precision. Cast the normalized values back to the input dtype before multiplying by `weight`, and return the final result in the input dtype. The implementation should handle two-dimensional and three-dimensional inputs, odd feature sizes, negative values, float32 and float16 tensors, and non-default `eps` values.
