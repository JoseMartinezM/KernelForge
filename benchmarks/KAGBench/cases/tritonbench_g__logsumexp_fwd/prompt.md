# logsumexp_fwd

Implement `logsumexp_fwd.py` with this public API:

```python
from typing import Optional
import torch


def logsumexp_fwd(
    x: torch.Tensor,
    scale: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    ...
```

`x` is a contiguous CUDA tensor. The function returns the log-sum-exp reduction over the last dimension of `x`, so the output shape is `x.shape[:-1]`.

Behavior requirements:

- If `scale` is provided, multiply `x` by that scalar before reducing.
- Match the numerically stable behavior of `torch.logsumexp(..., dim=-1)`.
- The default result dtype is `torch.float32`, including for lower-precision inputs.
- If `dtype` is provided and is not `torch.float`/`torch.float32`, cast the final result to `dtype`.
- Do not modify the input tensor in place.
