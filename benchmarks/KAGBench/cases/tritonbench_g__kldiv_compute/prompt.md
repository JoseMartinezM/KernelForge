# KL divergence element

Implement `kldivergence(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor`.

For two strictly positive CUDA tensors `x` and `y` with the same shape, return the elementwise value:

```python
x * torch.log(x / y)
```

The returned tensor must have the same shape and dtype as `x`. Inputs used by the tests are contiguous floating-point tensors on CUDA and have matching shapes.
