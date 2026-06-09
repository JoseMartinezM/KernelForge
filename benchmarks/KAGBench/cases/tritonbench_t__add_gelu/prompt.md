Implement `add_gelu(input, other, alpha=1, approximate='none', out=None)`.

The function adds `input` to `alpha * other`, where `other` may be a tensor or a Python number, then applies the Gaussian Error Linear Unit activation to the sum. Support `approximate='none'` and `approximate='tanh'` with the same numerical behavior as PyTorch's GELU.

If `out` is supplied, write the computed result into `out` and return `out`. Otherwise, return the computed tensor. Inputs are CUDA tensors, and normal PyTorch broadcasting rules for `input` and `other` should apply.
