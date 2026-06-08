# Log softmax

Implement `log_softmax(x: torch.Tensor, dim=-1, dtype=None) -> torch.Tensor`.

The function receives a CUDA floating-point tensor `x` and returns `torch.nn.functional.log_softmax(x, dim=dim, dtype=dtype)`. The output must have the same shape as `x`; when `dtype` is `None`, it should have the same dtype as `x`, and when `dtype` is provided, it should use that dtype.

Support positive and negative dimensions, including reductions over dimensions other than the last one. The implementation should be numerically stable for inputs containing large positive or negative values. Do not modify `x` in place.

The operation must participate in PyTorch autograd: gradients of scalar losses derived from the output should match PyTorch for the tested floating-point inputs.