# sin_kernel

Implement `sin_kernel.py` with the following public API:

```python
def call_kernel(x: torch.Tensor) -> torch.Tensor:
    ...
```

`call_kernel` receives a contiguous CUDA floating-point tensor `x` and returns a new tensor containing the sine of every element of `x`, matching `torch.sin(x)`.

Requirements:

- Preserve the input shape, dtype, and device.
- Support one-dimensional and higher-rank contiguous tensors.
- Support `torch.float32` and `torch.float16` inputs.
- Handle arbitrary element counts, including non-power-of-two lengths and empty tensors.
- Do not modify `x` in place.
