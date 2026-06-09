Implement `fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-05, *, out=None)`.

The function computes `input1 + input2`, multiplies that result elementwise by `input2`, then applies group normalization using `num_groups`, `weight`, `bias`, and `eps`.

Use the same tensor compatibility rules as PyTorch for addition and multiplication. The normalized tensor should have NCHW layout, with channel count `C` divisible by `num_groups`; `weight` and `bias`, when provided, have shape `(C,)`.

If `out` is provided, copy the result into `out` and return `out`. Otherwise, return the newly computed tensor.
