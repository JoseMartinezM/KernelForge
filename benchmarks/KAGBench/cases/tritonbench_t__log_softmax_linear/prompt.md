Implement `log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None)`.

The function applies a fully connected linear transform followed by log-softmax:

- `input` has trailing feature dimension `in_features` and may have any leading dimensions.
- `weight` has shape `(out_features, in_features)`.
- If `bias` is provided, add it to the linear output using normal PyTorch broadcasting for a one-dimensional output-feature bias.
- Return the log-softmax of the linear output along `dim`.
- If `dtype` is provided, it controls the dtype used by the log-softmax result, matching `torch.nn.functional.log_softmax` behavior.

The public API must be exactly the function above and must accept the same positional and keyword arguments.
