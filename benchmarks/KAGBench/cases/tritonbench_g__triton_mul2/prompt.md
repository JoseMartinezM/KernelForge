# Multiply tensor values by two

Implement the following functions in `triton_mul2.py`:

- `triton_mul2(x: torch.Tensor, BLOCK_SIZE: int = 16) -> torch.Tensor`
- `triton_mul2_inplace(x: torch.Tensor, BLOCK_SIZE: int = 16) -> torch.Tensor`

Both functions receive a CUDA tensor `x`. `BLOCK_SIZE` is a tunable integer argument in the public signature; correct results should be equivalent for different supported values.

Expected behavior:

- `triton_mul2` returns a new tensor with the same shape, dtype, and device as `x`, with every element equal to `2 * x`.
- `triton_mul2` must not modify `x`.
- `triton_mul2_inplace` doubles every element of `x` in place and returns `x`.
- Support contiguous floating point CUDA tensors of any rank.
