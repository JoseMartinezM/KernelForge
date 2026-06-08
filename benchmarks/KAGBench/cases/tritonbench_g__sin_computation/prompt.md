# sin_computation

Implement `sin_computation.py` with the following public API:

```python
def sin_triton(x: torch.Tensor, out: torch.Tensor):
    ...
```

`sin_triton` receives two same-shaped contiguous CUDA floating tensors. It must write the elementwise sine of `x` into the provided `out` tensor.

Requirements:

- Mutate `out` in place; callers will inspect `out` after the function returns.
- The return value is not important.
- Do not modify `x`.
- Preserve `out`'s shape and dtype.
- Support contiguous floating point CUDA tensors, including one-dimensional and higher-rank shapes.
- The result should match `torch.sin(x)` elementwise.
