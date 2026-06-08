# Row-wise softmax

Implement `triton_softmax(x: torch.Tensor) -> torch.Tensor`.

The function receives a two-dimensional contiguous CUDA floating-point tensor `x`. Return a tensor with the same shape and dtype as `x` where each row is normalized independently with softmax over the second dimension:

`output[i, j] = exp(x[i, j]) / sum_j exp(x[i, j])`

The implementation should be numerically stable for rows containing large positive or negative values. Do not modify `x` in place.
