# Matrix-vector multiplication

Implement `mv(inp: torch.Tensor, vec: torch.Tensor) -> torch.Tensor`.

The function receives a two-dimensional CUDA matrix `inp` and a one-dimensional CUDA vector `vec`. The vector length equals `inp.shape[1]`.

Return the matrix-vector product with shape `(inp.shape[0],)`, matching `torch.mv(inp, vec)` for valid inputs. The result should be on CUDA and use the same dtype behavior as PyTorch for the input tensors. Valid strided CUDA tensors accepted by `torch.mv` should also be handled.
