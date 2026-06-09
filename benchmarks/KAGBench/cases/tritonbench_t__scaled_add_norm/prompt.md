Implement `scaled_add_norm(y, x, alpha)`.

The function receives two one-dimensional CUDA tensors `y` and `x` with the same shape, plus a Python numeric scalar `alpha`. It must update `y` in place by adding `alpha * x` elementwise, then return the Euclidean norm of the updated `y` as a scalar tensor.

The public API is:

```python
def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    ...
```

The returned value and the final contents of `y` should match PyTorch behavior for `y += alpha * x` followed by `torch.norm(y)`.
