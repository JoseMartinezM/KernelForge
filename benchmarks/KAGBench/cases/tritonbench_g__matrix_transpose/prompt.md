# Matrix transpose

Implement the public function `wrapper(size_m, d_head)` in `matrix_transpose.py`.

## API

```python
def wrapper(size_m: int, d_head: int) -> torch.Tensor:
    ...
```

## Behavior

- Create a CUDA tensor named conceptually as the input matrix with shape `(size_m, d_head)`, dtype `torch.float16`, and values sampled from `torch.randn`.
- Return its transpose as a CUDA tensor with shape `(d_head, size_m)` and dtype `torch.float16`.
- The function should respect PyTorch's global random seed. If callers set `torch.manual_seed(...)` immediately before calling `wrapper`, the generated values and returned transpose should be deterministic.
- The returned tensor values must match the PyTorch reference implementation for the same seed, shape, device, and dtype.
