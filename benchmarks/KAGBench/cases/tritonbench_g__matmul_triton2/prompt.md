# Matrix multiplication

Implement the following public API in `matmul_triton2.py`:

```python
def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ...
```

`a` and `b` are two-dimensional CUDA tensors with compatible matrix multiplication dimensions: `a.shape[1] == b.shape[0]`. The tensors have matching dtypes.

Return the matrix product `a @ b` with shape `(a.shape[0], b.shape[1])`. The result must be on CUDA and have the same dtype as `a`.

Inputs used by the tests are contiguous, and may include square or rectangular matrices, odd and non-power-of-two dimensions, negative values, `float32`, and `float16` tensors.
