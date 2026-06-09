Implement `fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None)`.

The function computes a matrix-vector product, applies sigmoid, then subtracts `other` scaled by `alpha`:

1. Compute `torch.mv(input, vec)`.
2. Apply `torch.sigmoid` to that vector.
3. Compute `torch.sub(sigmoid_result, other, alpha=alpha)`.
4. If `out` is provided, copy the result into `out` and return `out`; otherwise return the newly computed result.

Public API:
- `input`: a 2D CUDA tensor of shape `(n, m)`.
- `vec`: a 1D CUDA tensor of shape `(m,)`.
- `other`: a CUDA tensor or Python number broadcastable to shape `(n,)`.
- `alpha`: number scaling `other` in the subtraction.
- `out`: optional CUDA tensor of shape `(n,)` receiving the result.

Your implementation should match PyTorch semantics for supported floating point inputs.
