# Dropout with an explicit keep mask

Implement the following public API:

```python
def dropout(x: torch.Tensor, x_keep: torch.Tensor, p: float) -> torch.Tensor:
    ...
```

`x` is a contiguous CUDA tensor. `x_keep` is a contiguous CUDA tensor with the same shape as `x`; entries that convert to `True` indicate elements to keep, and entries that convert to `False` indicate elements to drop. `p` is the dropout probability and will be in the half-open interval `[0, 1)`.

Return a tensor with the same shape and dtype as `x` where kept elements are scaled by the inverted-dropout factor `1 / (1 - p)` and dropped elements are zero:

```python
torch.where(x_keep.bool(), x / (1 - p), 0)
```

The operation is elementwise and should preserve the input shape for one-dimensional and higher-rank tensors.
