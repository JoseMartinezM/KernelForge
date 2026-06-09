Implement `softmax_mul(input, other, dim, dtype=None, out=None)`.

The function computes `torch.nn.functional.softmax(input, dim=dim, dtype=dtype)` and multiplies the result by `other`.

Parameters:
- `input`: a PyTorch tensor.
- `other`: a compatible same-shape PyTorch tensor or a Python numeric scalar.
- `dim`: the dimension over which to compute softmax.
- `dtype`: optional data type passed to the softmax computation.
- `out`: optional output tensor. If provided, copy the computed result into `out` and return `out` itself.

Return the multiplied softmax result, or the provided `out` tensor after it has been filled.
