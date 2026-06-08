# Vector Addition

Implement `vector_addition.py` with the following public API:

```python
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ...
```

`add` receives two CUDA tensors with the same shape and dtype and returns a new tensor containing the elementwise sum `x + y`. The returned tensor should have the same shape and dtype as `x`.

The function is expected to handle one-dimensional CUDA tensors of varying lengths, including lengths that are not powers of two. It should support both `torch.float32` and `torch.float16` inputs. Inputs used by the tests are contiguous tensors.
