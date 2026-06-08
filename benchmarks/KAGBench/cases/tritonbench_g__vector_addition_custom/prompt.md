# Vector Addition Custom

Implement `vector_addition_custom.py` with the following public API:

```python
import torch


def custom_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ...
```

`a` and `b` are same-length one-dimensional contiguous CUDA tensors with the same dtype. Return a new tensor containing the elementwise sum `a + b`. The returned tensor must have the same shape and dtype as `a`.

The function should handle vectors of varying lengths, including odd lengths, lengths that are not powers of two, and empty tensors. The tests use matching contiguous vector inputs and include both `torch.float32` and `torch.float16` values.
