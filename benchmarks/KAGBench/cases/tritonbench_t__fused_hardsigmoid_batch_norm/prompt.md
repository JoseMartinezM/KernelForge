Implement `fused_hardsigmoid_batch_norm(x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05, inplace=False)`.

The function should apply PyTorch batch normalization to `x` using the supplied running statistics, optional affine parameters, `training`, `momentum`, and `eps` arguments. It should then apply PyTorch hard-sigmoid activation to the normalized result, honoring the `inplace` argument for that activation, and return the activated tensor.

The public API and argument defaults must match the signature above. Inputs are CUDA tensors whose channel dimension is dimension 1, so `running_mean`, `running_var`, `weight`, and `bias` each have length equal to `x.shape[1]` when provided. Support typical PyTorch batch-normalization input ranks such as `(N, C)` and `(N, C, ...)`.
