# ReLU CUDA Kernel

Implement the public function `relu(x)`.

## API

```python
def relu(x: torch.Tensor) -> torch.Tensor:
    ...
```

- `x` is a CUDA tensor containing floating-point values.
- Return a CUDA tensor with the same shape as `x`.
- The returned tensor must have dtype `torch.float32`.
- Each output element is the rectified linear unit of the corresponding input element: values less than zero become `0`, while zero and positive values are preserved.
- The function should work for one-dimensional tensors and higher-rank tensors accepted by normal PyTorch elementwise operations.
