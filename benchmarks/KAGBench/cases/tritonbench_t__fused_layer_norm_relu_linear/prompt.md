Implement `fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None, eps=1e-05, elementwise_affine=True)`.

The function applies these operations in order:

1. Compute a linear transformation of `input` using `weight` and optional `bias`, matching `torch.nn.functional.linear`.
2. Apply ReLU to the linear result.
3. Apply layer normalization to the ReLU result with `normalized_shape` and `eps`.

`input` has shape `(*, in_features)`, `weight` has shape `(out_features, in_features)`, and `bias`, when provided, has shape `(out_features,)`. If `normalized_shape` is an integer, treat it as a one-element shape. Return the normalized tensor.

The `elementwise_affine` parameter is accepted for signature compatibility; this function does not use separate layer-normalization scale or bias parameters, so changing `elementwise_affine` does not alter the result.
