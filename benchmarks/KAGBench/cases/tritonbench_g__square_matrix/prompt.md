# Square Matrix

Implement the following public API in `square_matrix.py`:

```python
def square(x: torch.Tensor) -> torch.Tensor:
    ...
```

`x` is a two-dimensional CUDA tensor. Return a new tensor containing the
elementwise square of `x` (`x * x`). The output must preserve the input shape,
dtype, and device, and the input tensor must not be modified.

Inputs used by the tests are contiguous, and may include non-square matrices,
odd dimensions, non-power-of-two dimensions, negative values, `float32`, and
`float16` tensors.
