Implement `tanh_linear(input, weight, bias=None)`.

The function applies a fully connected linear transformation followed by a hyperbolic tangent activation:

1. Multiply `input` by the transpose of `weight`.
2. If `bias` is provided, add it to the linear result using normal PyTorch broadcasting rules.
3. Return `torch.tanh` of the result.

Public API:
- `input`: a CUDA tensor whose last dimension is `in_features`.
- `weight`: a CUDA tensor of shape `(out_features, in_features)`.
- `bias`: optional CUDA tensor broadcastable to the output shape, typically shape `(out_features,)`.
- Return a CUDA tensor with shape `input.shape[:-1] + (out_features,)`.

Your implementation should match PyTorch semantics for supported floating point inputs.
