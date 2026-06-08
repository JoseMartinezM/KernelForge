# Max reduction

Implement two public functions:

```python
def max(inp: torch.Tensor) -> torch.Tensor:
    ...

def max_dim(inp: torch.Tensor, dim=None, keepdim=False):
    ...
```

Inputs are contiguous CUDA tensors.

API requirements:

- `max(inp)` returns a scalar tensor containing the maximum value over all elements of `inp`.
- `max_dim(inp, dim, keepdim=False)` reduces `inp` along a single dimension and returns an object compatible with `torch.max(inp, dim=dim, keepdim=keepdim)`.
- The `max_dim` return value must expose `.values` and `.indices`; tuple-style `(values, indices)` returns are also acceptable if they contain the same tensors.
- Negative dimensions should be interpreted in the standard PyTorch way.
- If `keepdim` is `False`, the reduced dimension is removed from `.values` and `.indices`.
- If `keepdim` is `True`, the reduced dimension remains with size 1.
- Returned values should keep the input dtype and device. Returned indices should be `torch.int64` CUDA tensors.
- Test inputs avoid tied maxima so the expected indices are deterministic.

Your implementation should expose both public functions with the signatures above.
