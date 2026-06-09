Implement `sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)`.

The function should apply PyTorch batch normalization to `input` using the supplied running statistics, optional affine parameters, `training`, `momentum`, and `eps` arguments. It should then apply the sigmoid activation elementwise to the normalized result and return the activated tensor.

The public API and argument defaults must match the signature above. Inputs are CUDA tensors whose channel dimension is dimension 1, so `running_mean`, `running_var`, `weight`, and `bias` each have length equal to `input.shape[1]` when provided. Support typical PyTorch batch-normalization input ranks such as `(N, C)` and `(N, C, ...)`.
