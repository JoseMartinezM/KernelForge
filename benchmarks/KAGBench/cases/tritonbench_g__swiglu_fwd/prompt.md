Implement the public function `_swiglu_fwd(xy, out=None)` in `swiglu_fwd.py`.

The input `xy` is a CUDA tensor whose last dimension contains two equally sized halves. Split it into `x` and `y` along the last dimension and return the SwiGLU forward result:

```python
x * sigmoid(x) * y
```

The returned tensor must have the same batch dimensions as `xy` and a final dimension equal to half of `xy.shape[-1]`. For example, an input of shape `(B, 2 * N)` returns shape `(B, N)`, and an input of shape `(A, B, 2 * N)` returns shape `(A, B, N)`.

API requirements:

- Function name: `_swiglu_fwd`.
- Arguments:
  - `xy`: CUDA tensor with an even final dimension.
  - `out` (optional): output tensor. If provided, write the result into this tensor and return it reshaped to the batch shape plus half the final dimension.
- Preserve the mathematical behavior of PyTorch's `x * torch.sigmoid(x) * y`.
- Support `float32` inputs. Lower precision CUDA dtypes such as `float16`/`bfloat16` may be tested with appropriate tolerances.
- Inputs whose final dimension is not stored contiguously should still produce the correct result.
