Implement `softplus_linear(input, weight, bias=None, beta=1, threshold=20)`.

The function applies a fully connected linear transform followed by Softplus:

- `input` has trailing feature dimension `in_features` and may have any leading dimensions accepted by PyTorch linear operations.
- `weight` has shape `(out_features, in_features)`.
- If `bias` is provided, include it as the output-feature bias.
- Apply Softplus to the linear output using the supplied `beta` and `threshold` values, matching `torch.nn.functional.softplus` behavior.

The public API must be exactly the function above and must accept the same positional and keyword arguments.
