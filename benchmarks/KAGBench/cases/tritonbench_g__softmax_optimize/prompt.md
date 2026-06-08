# Row-wise softmax

Implement `softmax(x: torch.Tensor) -> torch.Tensor`.

`x` is a two-dimensional contiguous CUDA floating-point tensor. Return a tensor with the same shape and dtype as `x` containing row-wise softmax over the second dimension (`dim=1`):

`output[i, j] = exp(x[i, j]) / sum_k exp(x[i, k])`

The implementation should be numerically stable for rows with large positive or negative values. Do not modify `x` in place.
