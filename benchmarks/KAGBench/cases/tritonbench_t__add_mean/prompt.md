Implement `add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None)`.

The function adds `input` to `alpha * other`, then returns the mean of that sum. `other` may be a tensor or a Python numeric scalar. When `other` is a scalar, treat it as a value with the same device and data type as `input` before applying the arithmetic.

Parameters:
- `input`: a PyTorch tensor.
- `other`: a compatible same-shape PyTorch tensor or a Python numeric scalar.
- `dim`: `None`, an integer dimension, or a tuple of dimensions to reduce. With `None`, reduce all elements.
- `alpha`: numeric multiplier applied to `other` before addition.
- `keepdim`: whether reduced dimensions are retained with length 1.
- `dtype`: optional data type used by the mean computation.
- `out`: accepted for API compatibility; no special output-buffer behavior is required.

Return the same value that PyTorch would produce for `(input + alpha * other).mean(dim=dim, keepdim=keepdim, dtype=dtype)` with the scalar handling described above.
