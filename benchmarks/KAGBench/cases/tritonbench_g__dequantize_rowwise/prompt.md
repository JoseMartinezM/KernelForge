# Row-wise dequantization

Implement `dequantize_rowwise.py` with the following public API:

```python
def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor) -> torch.Tensor:
    ...
```

`x` is a two-dimensional CUDA tensor with dtype `torch.int8`. `state_x` is a one-dimensional CUDA floating-point tensor with one scale value per row, so `state_x.shape[0] == x.shape[0]`.

Return a CUDA tensor with the same shape as `x` and dtype `torch.float16`. Each output element is computed independently as:

```python
output[row, col] = state_x[row] * x[row, col] / 127
```

Do not modify `x` or `state_x` in place.
