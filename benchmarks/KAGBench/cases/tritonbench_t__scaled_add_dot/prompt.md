Implement `scaled_add_dot(y, x, alpha)`.

The function receives two one-dimensional CUDA tensors `y` and `x` with the same shape, plus a Python numeric scalar `alpha`. It must update `y` in place by adding `alpha * x` elementwise, then return the dot product of the updated `y` with itself as a scalar tensor.

The public API is:

```python
def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    ...
```

The returned value and the final contents of `y` should match PyTorch behavior for `y += alpha * x` followed by `torch.dot(y, y)`.
