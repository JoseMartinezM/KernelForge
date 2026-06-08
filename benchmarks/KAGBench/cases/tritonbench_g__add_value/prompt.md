# Add a constant value

Implement `puzzle1(x: torch.Tensor) -> torch.Tensor`.

The function receives a CUDA tensor `x` and returns a new tensor with the same shape and dtype where every element is increased by `10`.

Expected behavior:

- Do not modify `x` in place.
- Preserve the input shape and dtype in the returned tensor.
- Support floating point CUDA tensors.
- The result should be equivalent to `x + 10` using PyTorch broadcasting of the scalar constant.
