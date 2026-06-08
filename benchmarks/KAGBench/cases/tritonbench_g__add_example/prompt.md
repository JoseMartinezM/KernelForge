# Task

Implement `add_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor`.

The function receives two same-shaped contiguous CUDA tensors and returns a new tensor containing their elementwise sum, equivalent to `x + y`.

Requirements:

- Preserve the input shape in the returned tensor.
- Preserve the input dtype in the returned tensor.
- Do not mutate either input tensor.
- Inputs can be assumed to be valid same-shaped contiguous CUDA tensors.
